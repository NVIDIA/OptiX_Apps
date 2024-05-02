#include "mdl_wrapper.h"
#include "shaders/config.h"
#include "inc/CompileResult.h"
#include "inc/CheckMacros.h"

#include <algorithm>

#include <iostream>


#ifdef __linux__
#include <dlfcn.h>
#endif



// MDL Material specific functions.

static std::string replace(const std::string& source, const std::string& from, const std::string& to)
{
    if (source.empty())
    {
        return source;
    }

    std::string result;
    result.reserve(source.length());

    std::string::size_type lastPos = 0;
    std::string::size_type findPos;

    while (std::string::npos != (findPos = source.find(from, lastPos)))
    {
        result.append(source, lastPos, findPos - lastPos);
        result.append(to);

        lastPos = findPos + from.length();
    }

    //result += source.substr(lastPos);
    result.append(source, lastPos, source.length() - lastPos);

    return result;
}


static std::string buildModuleName(const std::string& path)
{
    if (path.empty())
    {
        return path;
    }

    // Build an MDL name. This assumes the path starts with a backslash (or slash on Linux).
    std::string name = path;

#if defined(_WIN32)
    if (name[0] != '\\')
    {
        name = std::string("\\") + path;
    }
    name = replace(name, "\\", "::");
#elif defined(__linux__)
    if (name[0] != '/')
    {
        name = std::string("/") + path;
    }
    name = replace(name, "/", "::");
#endif

    return name;
}


static std::string add_missing_material_signature(const mi::neuraylib::IModule* module,
                                           const std::string& material_name)
{
    // Return input if it already contains a signature.
    if (material_name.back() == ')')
    {
        return material_name;
    }

    mi::base::Handle<const mi::IArray> result(module->get_function_overloads(material_name.c_str()));

    // Not supporting multiple function overloads with the same name but different signatures.
    if (!result || result->get_length() != 1)
    {
        return std::string();
    }

    mi::base::Handle<const mi::IString> overloads(result->get_element<mi::IString>(static_cast<mi::Size>(0)));

    return overloads->get_c_str();
}


static bool isValidDistribution(mi::neuraylib::IExpression const* expr)
{
    if (expr == nullptr)
    {
        return false;
    }

    if (expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_constant(expr->get_interface<mi::neuraylib::IExpression_constant>());
        mi::base::Handle<mi::neuraylib::IValue const> value(expr_constant->get_value());

        if (value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
        {
            return false;
        }
    }

    return true;
}


// Returns a string-representation of the given message category
static const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind)
    {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
        return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
        return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
        return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
        return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
        return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
        return "Compiler DAG generator";
    default:
        break;
    }
    return "";
}

// Returns a string-representation of the given message severity
static const char* message_severity_to_string(mi::base::Message_severity severity)
{
    switch (severity)
    {
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "ERROR";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "WARNING";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "INFO";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "VERBOSE";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return "DEBUG";
    default:
        break;
    }
    return "";
}

class Default_logger: public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message(mi::base::Message_severity level,
                 const char* /* module_category */,
                 const mi::base::Message_details& /* details */,
                 const char* message) override
    {
        const char* severity = 0;

        switch (level)
        {
        case mi::base::MESSAGE_SEVERITY_FATAL:
            severity = "FATAL: ";
            MY_ASSERT(!"Default_logger() fatal error.");
            break;
        case mi::base::MESSAGE_SEVERITY_ERROR:
            severity = "ERROR: ";
            MY_ASSERT(!"Default_logger() error.");
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            severity = "WARN:  ";
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
            //return; // DEBUG No info messages.
            severity = "INFO:  ";
            break;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            return; // DEBUG No verbose messages.
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            return; // DEBUG No debug messages.
        case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
            return;
        }

        std::cerr << severity << message << '\n';
    }

    void message(mi::base::Message_severity level,
                 const char* module_category,
                 const char* message) override
    {
        this->message(level, module_category, mi::base::Message_details(), message);
    }
};


/// Callback that notifies the application about new resources when generating an
/// argument block for an existing target code.
class Resource_callback
    : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
{
public:
    /// Constructor.
    Resource_callback(mi::neuraylib::ITransaction* transaction,
                      mi::neuraylib::ITarget_code const* target_code,
                      Compile_result& compile_result)
        : m_transaction(mi::base::make_handle_dup(transaction))
        , m_target_code(mi::base::make_handle_dup(target_code))
        , m_compile_result(compile_result)
    {
    }

    /// Destructor.
    virtual ~Resource_callback() = default;

    /// Returns a resource index for the given resource value usable by the target code
    /// resource handler for the corresponding resource type.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    mi::Uint32 get_resource_index(mi::neuraylib::IValue_resource const* resource) override
    {
        // check whether we already know the resource index
        auto it = m_resource_cache.find(resource);
        if (it != m_resource_cache.end())
        {
            return it->second;
        }

        // handle resources already known by the target code
        mi::Uint32 res_idx = m_target_code->get_known_resource_index(m_transaction.get(), resource);
        if (res_idx != 0)
        {
            // only accept body resources
            switch (resource->get_kind())
            {
            case mi::neuraylib::IValue::VK_TEXTURE:
                if (m_target_code->get_texture_is_body_resource(res_idx))
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
                if (m_target_code->get_light_profile_is_body_resource(res_idx))
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                if (m_target_code->get_bsdf_measurement_is_body_resource(res_idx))
                    return res_idx;
                break;
            default:
                return 0u;  // invalid kind
            }
        }

        switch (resource->get_kind())
        {
        case mi::neuraylib::IValue::VK_TEXTURE:
        {
            mi::base::Handle<mi::neuraylib::IValue_texture const> val_texture(resource->get_interface<mi::neuraylib::IValue_texture const>());
            if (!val_texture)
            {
                return 0u;  // unknown resource
            }

            mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(val_texture->get_type());

            mi::neuraylib::ITarget_code::Texture_shape shape = mi::neuraylib::ITarget_code::Texture_shape(texture_type->get_shape());

            m_compile_result.textures.emplace_back(resource->get_value(), shape);
            res_idx = m_compile_result.textures.size() - 1;
            break;
        }

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            m_compile_result.light_profiles.emplace_back(resource->get_value());
            res_idx = m_compile_result.light_profiles.size() - 1;
            break;

        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            m_compile_result.bsdf_measurements.emplace_back(resource->get_value());
            res_idx = m_compile_result.bsdf_measurements.size() - 1;
            break;

        default:
            return 0u;  // invalid kind
        }

        m_resource_cache[resource] = res_idx;
        return res_idx;
    }

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    mi::Uint32 get_string_index(mi::neuraylib::IValue_string const* s) override
    {
        char const* str_val = s->get_value();
        if (str_val == nullptr)
            return 0u;

        for (mi::Size i = 0, n = m_target_code->get_string_constant_count(); i < n; ++i)
        {
            if (strcmp(m_target_code->get_string_constant(i), str_val) == 0)
            {
                return mi::Uint32(i);
            }
        }

        // string not known by code
        return 0u;
    }

private:
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;

    std::map<mi::neuraylib::IValue_resource const*, mi::Uint32> m_resource_cache;
    Compile_result& m_compile_result;
};



mi::neuraylib::INeuray* MdlWrapper::load_and_get_ineuray(const char* filename)
{
    if (!filename)
    {
        //#ifdef IRAY_SDK
        //    filename = "libneuray" MI_BASE_DLL_FILE_EXT;
        //#else
        filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
        //#endif
    }

#ifdef MI_PLATFORM_WINDOWS

    HMODULE handle = LoadLibraryA(filename);
    //if (!handle)
    //{
    //  // fall back to libraries in a relative lib folder, relevant for install targets
    //  std::string fallback = std::string("../../../lib/") + filename;
    //  handle = LoadLibraryA(fallback.c_str());
    //}
    if (!handle)
    {
        DWORD error_code = GetLastError();
        std::cerr << "ERROR: LoadLibraryA(" << filename << ") failed with error code " << error_code << '\n';
        return 0;
    }

    void* symbol = GetProcAddress(handle, "mi_factory");
    if (!symbol)
    {
        DWORD error_code = GetLastError();
        std::cerr << "ERROR: GetProcAddress(handle, \"mi_factory\") failed with error " << error_code << '\n';
        return 0;
    }

#else // MI_PLATFORM_WINDOWS

    void* handle = dlopen(filename, RTLD_LAZY);
    //if (!handle)
    //{
    //  // fall back to libraries in a relative lib folder, relevant for install targets
    //  std::string fallback = std::string("../../../lib/") + filename;
    //  handle = dlopen(fallback.c_str(), RTLD_LAZY);
    //}
    if (!handle)
    {
        std::cerr << "ERROR: dlopen(" << filename << " , RTLD_LAZY) failed with error code " << dlerror() << '\n';
        return 0;
    }

    void* symbol = dlsym(handle, "mi_factory");
    if (!symbol)
    {
        std::cerr << "ERROR: dlsym(handle, \"mi_factory\") failed with error " << dlerror() << '\n';
        return 0;
    }

#endif // MI_PLATFORM_WINDOWS

    m_dso_handle = handle;

    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
    if (!neuray)
    {
        mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
        if (!version)
        {
            std::cerr << "ERROR: Incompatible library. Could not determine version.\n";
        }
        else
        {
            std::cerr << "ERROR: Library version " << version->get_product_version() << " does not match header version " << MI_NEURAYLIB_PRODUCT_VERSION_STRING << '\n';
        }
        return 0;
    }

    //#ifdef IRAY_SDK
    //  if (authenticate(neuray) != 0)
    //  {
    //    std::cerr << "ERROR: Iray SDK Neuray Authentication failed.\n";
    //    unload();
    //    return 0;
    //  }
    //#endif

    return neuray;
}


mi::Sint32 MdlWrapper::load_plugin(mi::neuraylib::INeuray* neuray, const char* path)
{
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    // Try loading the requested plugin before adding any special handling
    mi::Sint32 res = plugin_conf->load_plugin_library(path);
    if (res == 0)
    {
        //std::cerr << "load_plugin(" << path << ") succeeded.\n"; // DEBUG The logger prints this.
        return 0;
    }

// Special handling for freeimage in the open source release.
// In the open source version of the plugin we are linking against a dynamic vanilla freeimage library.
// In the binary release, you can download from the MDL website, freeimage is linked statically and
// thereby requires no special handling.
#if defined(MI_PLATFORM_WINDOWS) && defined(MDL_SOURCE_RELEASE)
    if (strstr(path, "nv_freeimage" MI_BASE_DLL_FILE_EXT) != nullptr)
    {
        // Load the freeimage (without nv_ prefix) first.
        std::string freeimage_3rd_party_path = replace(path, "nv_freeimage" MI_BASE_DLL_FILE_EXT, "freeimage" MI_BASE_DLL_FILE_EXT);
        HMODULE handle_tmp = LoadLibraryA(freeimage_3rd_party_path.c_str());
        if (!handle_tmp)
        {
            DWORD error_code = GetLastError();
            std::cerr << "ERROR: load_plugin(" << freeimage_3rd_party_path << " failed with error " << error_code << '\n';
        }
        else
        {
            std::cerr << "Pre-loading library " << freeimage_3rd_party_path << " succeeded\n";
        }

        // Try to load the plugin itself now
        res = plugin_conf->load_plugin_library(path);
        if (res == 0)
        {
            std::cerr << "load_plugin(" << path << ") succeeded.\n"; // DAR FIXME The logger prints this as info anyway.
            return 0;
        }
    }
#endif

    // return the failure code
    std::cerr << "ERROR: load_plugin(" << path << ") failed with error " << res << '\n';

    return res;
}


bool MdlWrapper::initMDL(const std::vector<std::string>& searchPaths)
{
    // Load MDL SDK library and create a Neuray handle.
    m_neuray = load_and_get_ineuray(nullptr);

    if (!m_neuray.is_valid_interface())
    {
        std::cerr << "ERROR: Initialization of MDL SDK failed: libmdl_sdk" MI_BASE_DLL_FILE_EXT " not found or wrong version.\n";
        return false;
    }

    // Create the MDL compiler.
    m_mdl_compiler = m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
    if (!m_mdl_compiler)
    {
        std::cerr << "ERROR: Initialization of MDL compiler failed.\n";
        return false;
    }

    // Configure Neuray.
    // m_mdl_config->set_logger() and get_logger() are deprecated inside the MDL SDK 2023-11-14
    m_logging_config = m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>();
    if (!m_logging_config)
    {
        std::cerr << "ERROR: Retrieving logging configuration failed.\n";
        return false;
    }
    m_logger = mi::base::make_handle(new Default_logger());
    m_logging_config->set_receiving_logger(m_logger.get());

    m_mdl_config = m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
    if (!m_mdl_config)
    {
        std::cerr << "ERROR: Retrieving MDL configuration failed.\n";
        return false;
    }

    // Convenient default search paths for the NVIDIA MDL vMaterials!

    // Environment variable MDL_SYSTEM_PATH.
    // Defaults to "C:\ProgramData\NVIDIA Corporation\mdl\" under Windows.
    // Required to find ::nvidia::core_definitions imports used inside the vMaterials *.mdl files.
    m_mdl_config->add_mdl_system_paths();

    // Environment variable MDL_USER_PATH.
    // Defaults to "C:\Users\<username>\Documents\mdl\" under Windows.
    // Required to find the vMaterials *.mdl files and their resources.
    m_mdl_config->add_mdl_user_paths();

    // Add all additional MDL and resource search paths defined inside the system description file as well.
    for (auto const& path : searchPaths)
    {
        mi::Sint32 result = m_mdl_config->add_mdl_path(path.c_str());
        if  (result != 0)
        {
            std::cerr << "WARNING: add_mdl_path( " << path << ") failed with " << result << '\n';
        }

        result = m_mdl_config->add_resource_path(path.c_str());
        if (result != 0)
        {
            std::cerr << "WARNING: add_resource_path( " << path << ") failed with " << result << '\n';
        }
    }

// Load plugins.
#if USE_OPENIMAGEIO_PLUGIN
    if (load_plugin(m_neuray.get(), "nv_openimageio" MI_BASE_DLL_FILE_EXT) != 0)
    {
        std::cerr << "FATAL: Failed to load nv_openimageio plugin\n";
        return false;
    }
#else
    if (load_plugin(m_neuray.get(), "nv_freeimage" MI_BASE_DLL_FILE_EXT) != 0)
    {
        std::cerr << "FATAL: Failed to load nv_freeimage plugin\n";
        return false;
    }
#endif

    if (load_plugin(m_neuray.get(), "dds" MI_BASE_DLL_FILE_EXT) != 0)
    {
        std::cerr << "FATAL: Failed to load dds plugin\n";
        return false;
    }

    if (m_neuray->start() != 0)
    {
        std::cerr << "FATAL: Starting MDL SDK failed.\n";
        return false;
    }

    m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();

    m_global_scope = m_database->get_global_scope();

    m_mdl_factory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();

    // Configure the execution context.
    // Used for various configurable operations and for querying warnings and error messages.
    // It is possible to have more than one, in order to use different settings.
    m_execution_context = m_mdl_factory->create_execution_context();

    m_execution_context->set_option("internal_space", "coordinate_world");  // equals default
    m_execution_context->set_option("bundle_resources", false);             // equals default
    m_execution_context->set_option("meters_per_scene_unit", 1.0f);         // equals default
    m_execution_context->set_option("mdl_wavelength_min", 380.0f);          // equals default
    m_execution_context->set_option("mdl_wavelength_max", 780.0f);          // equals default
    // If true, the "geometry.normal" field will be applied to the MDL state prior to evaluation of the given DF.
    m_execution_context->set_option("include_geometry_normal", true);       // equals default

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

    m_mdl_backend = mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX);

    // Hardcoded values!
    MY_STATIC_ASSERT(NUM_TEXTURE_SPACES == 1 || NUM_TEXTURE_SPACES == 2);
    // The renderer only supports one or two texture spaces.
    // The hair BSDF requires two texture coordinates!
    // If you do not use the hair BSDF, NUM_TEXTURE_SPACES should be set to 1 for performance reasons.

    if (m_mdl_backend->set_option("num_texture_spaces", std::to_string(NUM_TEXTURE_SPACES).c_str()) != 0)
    {
        return false;
    }

    if (m_mdl_backend->set_option("num_texture_results", std::to_string(NUM_TEXTURE_RESULTS).c_str()) != 0)
    {
        return false;
    }

    // Use SM 5.0 for Maxwell and above.
    if (m_mdl_backend->set_option("sm_version", "50") != 0)
    {
        return false;
    }

    if (m_mdl_backend->set_option("tex_lookup_call_mode", "direct_call") != 0)
    {
        return false;
    }

    // PERF Let expression functions return the result as value, instead of void return and sret pointer argument which is slower.
    if (m_mdl_backend->set_option("lambda_return_mode", "value") != 0)
    {
        return false;
    }

    //if (enable_derivatives) // == false. Not supported in this renderer
    //{
    //  // Option "texture_runtime_with_derivs": Default is disabled.
    //  // We enable it to get coordinates with derivatives for texture lookup functions.
    //  if (m_mdl_backend->set_option("texture_runtime_with_derivs", "on") != 0)
    //  {
    //    return false;
    //  }
    //}

    if (m_mdl_backend->set_option("inline_aggressively", "on") != 0)
    {
        return false;
    }

    // FIXME Determine what scene data the renderer needs to provide here.
    // FIXME scene_data_names is not a supported option anymore!
    //if (m_mdl_backend->set_option("scene_data_names", "*") != 0)
    //{
    //  return false;
    //}

    // PERF Disable code generation for distribution pdf functions.
    // The unidirectional light transport in this renderer never calls them.
    // The sample and evaluate functions return the necessary pdf values.
    if (m_mdl_backend->set_option("enable_pdf", "off") != 0)
    {
        std::cerr << "WARNING: MdlWrapper::initMDL() Setting backend option enable_pdf to off failed.\n";
        // Not a fatal error if this cannot be set.
        // return false;
    }

    m_image_api = m_neuray->get_api_component<mi::neuraylib::IImage_api>();

    return true;
}


void MdlWrapper::shutdownMDL()
{
    m_shaderConfigurations.clear();
    m_shaders.clear(); // Code handles must be destroyed or there will be memory leaks indicated by MDL.

    m_mapMaterialHashToShaderIndex.clear();

    m_image_api.reset();
    m_mdl_backend.reset();
    m_execution_context.reset();
    m_mdl_factory.reset();
    m_global_scope.reset();
    m_database.reset();
    m_mdl_config.reset();
    m_logging_config.reset();
    m_mdl_compiler.reset();

    m_neuray->shutdown();
    m_neuray = nullptr;
}


bool MdlWrapper::log_messages(mi::neuraylib::IMdl_execution_context* context)
{
    m_last_mdl_error.clear();

    for (mi::Size i = 0; i < context->get_messages_count(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        m_last_mdl_error += message_kind_to_string(message->get_kind());
        m_last_mdl_error += " ";
        m_last_mdl_error += message_severity_to_string(message->get_severity());
        m_last_mdl_error += ": ";
        m_last_mdl_error += message->get_string();
        m_last_mdl_error += "\n";
    }
    return context->get_error_messages_count() == 0;
}


// Query expressions inside the compiled material to determine which direct callable functions need to be generated and
// what the closest hit program needs to call to fully render this material.
void MdlWrapper::determineShaderConfiguration(const Compile_result& res, ShaderConfiguration& config)
{
    config.is_thin_walled_constant = false;
    config.thin_walled             = false;

    mi::base::Handle<mi::neuraylib::IExpression const> thin_walled_expr(res.compiled_material->lookup_sub_expression("thin_walled"));
    if (thin_walled_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        config.is_thin_walled_constant = true;

        mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(thin_walled_expr->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_bool const> value_bool(expr_const->get_value<mi::neuraylib::IValue_bool>());

        config.thin_walled = value_bool->get_value();
    }

    mi::base::Handle<mi::neuraylib::IExpression const> surface_scattering_expr(res.compiled_material->lookup_sub_expression("surface.scattering"));

    config.is_surface_bsdf_valid = isValidDistribution(surface_scattering_expr.get()); // True if surface.scattering != bsdf().

    config.is_backface_bsdf_valid = false;

    // The backface scattering is only used for thin-walled materials.
    if (!config.is_thin_walled_constant || config.thin_walled)
    {
        // When backface == bsdf() MDL uses the surface scattering on both sides, irrespective of the thin_walled state.
        mi::base::Handle<mi::neuraylib::IExpression const> backface_scattering_expr(res.compiled_material->lookup_sub_expression("backface.scattering"));

        config.is_backface_bsdf_valid = isValidDistribution(backface_scattering_expr.get()); // True if backface.scattering != bsdf().

        if (config.is_backface_bsdf_valid)
        {
            // Only use the backface scattering when it's valid and different from the surface scattering expression.
            config.is_backface_bsdf_valid = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) !=
                                             res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING));
        }
    }

    // Surface EDF.
    mi::base::Handle<mi::neuraylib::IExpression const> surface_edf_expr(res.compiled_material->lookup_sub_expression("surface.emission.emission"));

    config.is_surface_edf_valid = isValidDistribution(surface_edf_expr.get());

    config.is_surface_intensity_constant      = true;
    config.surface_intensity                  = mi::math::Color(0.0f, 0.0f, 0.0f);
    config.is_surface_intensity_mode_constant = true;
    config.surface_intensity_mode             = 0; // == intensity_radiant_exitance;

    if (config.is_surface_edf_valid)
    {
        // Surface emission intensity.
        mi::base::Handle<mi::neuraylib::IExpression const> surface_intensity_expr(res.compiled_material->lookup_sub_expression("surface.emission.intensity"));

        config.is_surface_intensity_constant = false;

        if (surface_intensity_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<mi::neuraylib::IExpression_constant const> intensity_const(surface_intensity_expr->get_interface<mi::neuraylib::IExpression_constant const>());
            mi::base::Handle<mi::neuraylib::IValue_color const> intensity_color(intensity_const->get_value<mi::neuraylib::IValue_color>());

            if (get_value(intensity_color.get(), config.surface_intensity) == 0)
            {
                config.is_surface_intensity_constant = true;
            }
        }

        // Surface emission mode. This is a uniform and normally the default intensity_radiant_exitance
        mi::base::Handle<mi::neuraylib::IExpression const> surface_intensity_mode_expr(res.compiled_material->lookup_sub_expression("surface.emission.mode"));

        config.is_surface_intensity_mode_constant = false;

        if (surface_intensity_mode_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(surface_intensity_mode_expr->get_interface<mi::neuraylib::IExpression_constant const>());
            mi::base::Handle<mi::neuraylib::IValue_enum const> value_enum(expr_const->get_value<mi::neuraylib::IValue_enum>());

            config.surface_intensity_mode = value_enum->get_value();

            config.is_surface_intensity_mode_constant = true;
        }
    }

    // Backface EDF.
    config.is_backface_edf_valid               = false;
    // DEBUG Is any of this needed at all or is the BSDF init() function handling all this?
    config.is_backface_intensity_constant      = true;
    config.backface_intensity                  = mi::math::Color(0.0f, 0.0f, 0.0f);
    config.is_backface_intensity_mode_constant = true;
    config.backface_intensity_mode             = 0; // == intensity_radiant_exitance;
    config.use_backface_edf                    = false;
    config.use_backface_intensity              = false;
    config.use_backface_intensity_mode         = false;

    // A backface EDF is only used on thin-walled materials with a backface.emission.emission != edf()
    if (!config.is_thin_walled_constant || config.thin_walled)
    {
        mi::base::Handle<mi::neuraylib::IExpression const> backface_edf_expr(res.compiled_material->lookup_sub_expression("backface.emission.emission"));

        config.is_backface_edf_valid = isValidDistribution(backface_edf_expr.get());

        if (config.is_backface_edf_valid)
        {
            // Backface emission intensity.
            mi::base::Handle<mi::neuraylib::IExpression const> backface_intensity_expr(res.compiled_material->lookup_sub_expression("backface.emission.intensity"));

            config.is_backface_intensity_constant = false;

            if (backface_intensity_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
            {
                mi::base::Handle<mi::neuraylib::IExpression_constant const> intensity_const(backface_intensity_expr->get_interface<mi::neuraylib::IExpression_constant const>());
                mi::base::Handle<mi::neuraylib::IValue_color const> intensity_color(intensity_const->get_value<mi::neuraylib::IValue_color>());

                if (get_value(intensity_color.get(), config.backface_intensity) == 0)
                {
                    config.is_backface_intensity_constant = true;
                }
            }

            // Backface emission mode. This is a uniform and normally the default intensity_radiant_exitance.
            mi::base::Handle<mi::neuraylib::IExpression const> backface_intensity_mode_expr(res.compiled_material->lookup_sub_expression("backface.emission.mode"));

            config.is_backface_intensity_mode_constant = false;

            if (backface_intensity_mode_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
            {
                mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(backface_intensity_mode_expr->get_interface<mi::neuraylib::IExpression_constant const>());
                mi::base::Handle<mi::neuraylib::IValue_enum const> value_enum(expr_const->get_value<mi::neuraylib::IValue_enum>());

                config.backface_intensity_mode = value_enum->get_value();

                config.is_backface_intensity_mode_constant = true;
            }

            // When surface and backface expressions are identical, reuse the surface expression to generate less code.
            config.use_backface_edf = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
                                       res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION));

            // If the surface and backface emission use different intensities then use the backface emission intensity.
            config.use_backface_intensity = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY) !=
                                             res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY));

            // If the surface and backface emission use different modes (radiant exitance vs. power) then use the backface emission intensity mode.
            config.use_backface_intensity_mode = (res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_MODE) !=
                                                  res.compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_MODE));
        }
    }

    config.is_ior_constant = true;
    config.ior             = mi::math::Color(1.0f, 1.0f, 1.0f);

    mi::base::Handle<mi::neuraylib::IExpression const> ior_expr(res.compiled_material->lookup_sub_expression("ior"));
    if (ior_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(ior_expr->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

        if (get_value(value_color.get(), config.ior) == 0)
        {
            config.is_ior_constant = true;
        }
    }
    else
    {
        config.is_ior_constant = false;
    }

    // FIXME This renderer currently only supports a single df::anisotropic_vdf() under the volume.scattering expression.
    // MDL 1.8 added fog_vdf() and there are also mixers and modifiers on VDFs, which, while valid expressions, won't be evaluated.
    mi::base::Handle<mi::neuraylib::IExpression const> volume_vdf_expr(res.compiled_material->lookup_sub_expression("volume.scattering"));

    config.is_vdf_valid = isValidDistribution(volume_vdf_expr.get());

    // Absorption coefficient. Can be used without valid VDF.
    config.is_absorption_coefficient_constant = true;  // Default to constant and no absorption.
    config.use_volume_absorption              = false; // If there is no abosorption, the absorption coefficient is constant zero.
    config.absorption_coefficient             = mi::math::Color(0.0f, 0.0f, 0.0f); // No absorption.

    mi::base::Handle<mi::neuraylib::IExpression const> volume_absorption_coefficient_expr(res.compiled_material->lookup_sub_expression("volume.absorption_coefficient"));

    if (volume_absorption_coefficient_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_absorption_coefficient_expr->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

        if (get_value(value_color.get(), config.absorption_coefficient) == 0)
        {
            config.is_absorption_coefficient_constant = true;

            if (config.absorption_coefficient[0] != 0.0f || config.absorption_coefficient[1] != 0.0f || config.absorption_coefficient[2] != 0.0f)
            {
                config.use_volume_absorption = true;
            }
        }
    }
    else
    {
        config.is_absorption_coefficient_constant = false;
        config.use_volume_absorption              = true;
    }

    // Scattering coefficient. Only used when there is a valid VDF.
    config.is_scattering_coefficient_constant = true; // Default to constant and no scattering. Assumes invalid VDF.
    config.use_volume_scattering              = false;
    config.scattering_coefficient             = mi::math::Color(0.0f, 0.0f, 0.0f); // No scattering

    // Directional bias (Henyey_Greenstein g factor.) Only used when there is a valid VDF and volume scattering coefficient not zero.
    config.is_directional_bias_constant = true;
    config.directional_bias             = 0.0f;

    // The anisotropic_vdf() is the only valid VDF.
    // The scattering_coefficient, directional_bias (and emission_intensity) are only needed when there is a valid VDF.
    if (config.is_vdf_valid)
    {
        mi::base::Handle<mi::neuraylib::IExpression const> volume_scattering_coefficient_expr(res.compiled_material->lookup_sub_expression("volume.scattering_coefficient"));

        if (volume_scattering_coefficient_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_scattering_coefficient_expr->get_interface<mi::neuraylib::IExpression_constant const>());
            mi::base::Handle<mi::neuraylib::IValue_color const> value_color(expr_const->get_value<mi::neuraylib::IValue_color>());

            if (get_value(value_color.get(), config.scattering_coefficient) == 0)
            {
                config.is_scattering_coefficient_constant = true;

                if (config.scattering_coefficient[0] != 0.0f || config.scattering_coefficient[1] != 0.0f || config.scattering_coefficient[2] != 0.0f)
                {
                    config.use_volume_scattering = true;
                }
            }
        }
        else
        {
            config.is_scattering_coefficient_constant = false;
            config.use_volume_scattering              = true;
        }

        // FIXME This assumes a single anisotropic_vdf() under the volume.scattering expression!
        // MDL 1.8 fog_vdf() or VDF mixers or modifiers are not supported, yet, and the returned expression will be nullptr then.
        mi::base::Handle<mi::neuraylib::IExpression const> volume_directional_bias_expr(res.compiled_material->lookup_sub_expression("volume.scattering.directional_bias"));

        // Ignore unsupported volume.scattering expressions, instead use anisotropic_vdf() with constant isotropic directional_bias == 0.0f.
        if (volume_directional_bias_expr.get() != nullptr)
        {
            if (volume_directional_bias_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
            {
                config.is_directional_bias_constant = true;

                mi::base::Handle<mi::neuraylib::IExpression_constant const> expr_const(volume_directional_bias_expr->get_interface<mi::neuraylib::IExpression_constant const>());
                mi::base::Handle<mi::neuraylib::IValue_float const> value_float(expr_const->get_value<mi::neuraylib::IValue_float>());

                // 0.0f is isotropic. No need to distinguish. The sampleHenyeyGreenstein() function takes this as parameter anyway.
                config.directional_bias = value_float->get_value();
            }
            else
            {
                config.is_directional_bias_constant = false;
            }
        }
        else
        {
            std::cerr << "WARNING: Unsupported volume.scattering expression. directional_bias not found, using isotropic VDF.\n";
        }

        // volume.scattering.emission_intensity is not supported by this renderer.
        // Also the volume absorption and scattering coefficients are assumed to be homogeneous in this renderer.
    }

    // geometry.displacement is not supported by this renderer.

    // geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

    config.cutout_opacity             = 1.0f; // Default is fully opaque.
    config.is_cutout_opacity_constant = res.compiled_material->get_cutout_opacity(&config.cutout_opacity); // This sets cutout opacity to -1.0 when it's not constant!
    config.use_cutout_opacity         = !config.is_cutout_opacity_constant || config.cutout_opacity < 1.0f;

    mi::base::Handle<mi::neuraylib::IExpression const> hair_bsdf_expr(res.compiled_material->lookup_sub_expression("hair"));

    config.is_hair_bsdf_valid = isValidDistribution(hair_bsdf_expr.get()); // True if hair != hair_bsdf().
}


void MdlWrapper::initMaterialsMDL(std::vector<MaterialMDL*>& materialsMDL, std::vector<Device*>& devices_active)
{
    // This will compile the material to OptiX PTX code and build the OptiX program and texture data on all devices
    // and track the material configuration and parameters stored inside the Application class to be able to build the GUI.
    for (MaterialMDL* materialMDL : materialsMDL)
    {
        initMaterialMDL(materialMDL, devices_active);
    }

    // After all MDL material references have been handled and the device side data has been allocated, upload the necessary data to the GPU.
    for (size_t i = 0; i < devices_active.size(); ++i)
    {
        devices_active[i]->initTextureHandler(materialsMDL);
    }
}


void MdlWrapper::initMaterialMDL(MaterialMDL* material, std::vector<Device*>& devices_active)
{
    // This function is called per unique material reference.
    // No need to check for duplicate reference definitions.
    printf("  point A\n");

    mi::base::Handle<mi::neuraylib::ITransaction> handleTransaction = mi::base::make_handle<mi::neuraylib::ITransaction>(m_global_scope->create_transaction());
    mi::neuraylib::ITransaction* transaction = handleTransaction.get();

    // Local scope for all handles used inside the Compile_result.
    {
        Compile_result res;

        printf("  point B\n");
        // Split into separate functions to make the Neuray handles and transaction scope lifetime handling automatic.
        // When the function was successful, the Compile_result contains all information required to setup the device resources.
        const bool valid = compileMaterial(transaction, material, res);
        if (!valid)
        {
            std::cerr << "ERROR: MdlWrapper::initMaterialMDL() compileMaterial() failed. Material invalid.\n";
        }

        material->setIsValid(valid);
        printf("  point C\n");

        if (valid)
        {
            printf("  point D\n");
            // Create the OptiX programs on all devices.
            for (size_t device = 0; device < devices_active.size(); ++device)
            {
                devices_active[device]->compileMaterial(transaction, material, res, m_shaderConfigurations[material->getShaderIndex()]);
            }

            // Prepare 2D and 3D textures.

            // Create the CUDA texture arrays on the devices with peer-to-peer sharing when enabled.
            for (mi::Size idxRes = 1; idxRes < res.textures.size(); ++idxRes) // The zeroth index is the invalid texture.
            {
                bool first = true; // Only append each texture index to the MaterialMDL m_indicesToTextures vector once.

                {
                    for (size_t device = 0; device < devices_active.size(); ++device)
                    {
                        const TextureMDLHost* texture = devices_active[device]->prepareTextureMDL(transaction,
                                                                                                   m_image_api,
                                                                                                   res.textures[idxRes].db_name.c_str(),
                                                                                                   res.textures[idxRes].shape);
                        if (texture == nullptr)
                        {
                            std::cerr << "ERROR: initMaterialMDL(): prepareTextureMDL() failed for " << res.textures[idxRes].db_name << '\n';
                        }
                        else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
                        {
                            material->m_indicesToTextures.push_back(texture->m_index);
                        }
                    }
                }
            }

            // Prepare Bsdf_measurements.
            for (mi::Size idxRes = 1; idxRes < res.bsdf_measurements.size(); ++idxRes) // The zeroth index is the invalid Bsdf_measurement.
            {
                bool first = true;

                {
                    for (size_t device = 0; device < devices_active.size(); ++device)
                    {
                        const MbsdfHost* mbsdf = devices_active[device]->prepareMBSDF(transaction, res.target_code.get(), idxRes);

                        if (mbsdf == nullptr)
                        {
                            std::cerr << "ERROR: initMaterialMDL(): prepareMBSDF() failed for BSDF measurement " << idxRes << '\n';
                        }
                        else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
                        {
                            material->m_indicesToMBSDFs.push_back(mbsdf->m_index);
                        }
                    }
                }
            }

            // Prepare Light_profiles.
            for (mi::Size idxRes = 1; idxRes < res.light_profiles.size(); ++idxRes) // The zeroth index is the invalid light profile.
            {
                bool first = true;

                {
                    for (size_t device = 0; device < devices_active.size(); ++device)
                    {
                        const LightprofileHost* profile = devices_active[device]->prepareLightprofile(transaction, res.target_code.get(), idxRes);

                        if (profile == nullptr)
                        {
                            std::cerr << "ERROR: initMaterialMDL(): prepareLightprofile() failed for light profile " << idxRes << '\n';
                        }
                        else if (device == 0) // Only store the index once into the vector at the MaterialMDL.
                        {
                            material->m_indicesToLightprofiles.push_back(profile->m_index);
                        }
                    }
                }
            }
        }
    }

    transaction->commit();
}


bool MdlWrapper::compileMaterial(mi::neuraylib::ITransaction* transaction, MaterialMDL* materialMDL, Compile_result& res)
{
    // Build the fully qualified MDL module name.
    // The *.mdl file path has been converted to the proper OS format during input.
    std::string path = materialMDL->getPath();
    printf("        compiling material @ %s\n", path.c_str());

    // Path needs to end with ".mdl" so any path with 4 or less characters cannot be a valid path name.
    if (path.size() <= 4)
    {
        std::cerr << "ERROR: compileMaterial() Path name " << path << " too short.\n";
        return false;
    }

    const std::string::size_type last = path.size() - 4;

    if (path.substr(last, path.size()) != std::string(".mdl"))
    {
        std::cerr << "ERROR: compileMaterial() Path name " << path << " not matching \".mdl\".\n";
        return false;
    }

    std::string module_name = buildModuleName(path.substr(0, last));

    // Get everything to load the module.
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(m_neuray->get_api_component<mi::neuraylib::IMdl_factory>()); // FIXME Redundant, could use m_mdl_factory.

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    // Create an execution context for options and error message handling
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(mdl_factory->create_execution_context());

    mi::Sint32 reason = mdl_impexp_api->load_module(transaction, module_name.c_str(), context.get());
    if (reason < 0)
    {
        std::cerr << "ERROR: compileMaterial() Failed to load module " << module_name << '\n';
        switch (reason)
        {
        // case 1: // Success (module exists already, loading from file was skipped).
        // case 0: // Success (module was actually loaded from file).
        case -1:
            std::cerr << "The module name is invalid or a NULL pointer.\n";
            break;
        case -2:
            std::cerr << "Failed to find or to compile the module.\n";
            break;
        case -3:
            std::cerr << "The database name for an imported module is already in use but is not an MDL module,\n";
            std::cerr << "or the database name for a definition in this module is already in use.\n";
            break;
        case -4:
            std::cerr << "Initialization of an imported module failed.\n";
            break;
        default:
            std::cerr << "Unexpected return value " << reason << " from IMdl_impexp_api::load_module().\n";
            MY_ASSERT(!"Unexpected return value from IMdl_compiler::load_module()");
            break;
        }
    }

    if (!log_messages(context.get()))
    {
        return false;
    }

    // Get the database name for the module we loaded.
    mi::base::Handle<const mi::IString> module_db_name(mdl_factory->get_db_module_name(module_name.c_str()));

    // Note that the lifetime of this module handle must end before the transaction->commit() or there will be warnings.
    // This is automatically handled by placing the transaction into the caller.
    mi::base::Handle<const mi::neuraylib::IModule> module(transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
    {
        std::cerr << "ERROR: compileMaterial() Failed to access the loaded module " << module_db_name->get_c_str() << '\n';
        return false;
    }

    // Build the fully qualified data base name of the material.
    const std::string material_simple_name = materialMDL->getName();

    std::string material_db_name = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;

    material_db_name = add_missing_material_signature(module.get(), material_db_name);

    if (material_db_name.empty())
    {
        std::cerr << "ERROR: compileMaterial() Failed to find the material " + material_simple_name + " in the module " + module_name + ".\n";
        return false;
    }


    // Compile the material.

    // Create a material instance from the material definition with the default arguments.
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
    {
        std::cerr << "ERROR: compileMaterial() Material definition could not be created for " << material_simple_name << '\n';
        return false;
    }

    mi::Sint32 ret = 0;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(material_definition->create_function_call(0, &ret));
    if (ret != 0)
    {
        std::cerr << "ERROR: compileMaterial() Instantiating material " + material_simple_name + " failed";
        return false;
    }

    // Create a compiled material.
    // DEBUG Experiment with instance compilation as well to see how the performance changes.
    mi::Uint32 flags = mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;

    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance2(material_instance->get_interface<mi::neuraylib::IMaterial_instance>());

    res.compiled_material = material_instance2->create_compiled_material(flags, context.get());
    if (!log_messages(context.get()))
    {
        std::cerr << "ERROR: compileMaterial() create_compiled_material() failed.\n";
        return false;
    }

    // Check if the target code has already been generated for another material reference name and reuse the existing target code.
    int indexShader = -1; // Invalid index.

    mi::base::Uuid material_hash = res.compiled_material->get_hash();

    std::map<mi::base::Uuid, int>::const_iterator it = m_mapMaterialHashToShaderIndex.find(material_hash);
    if (it != m_mapMaterialHashToShaderIndex.end())
    {
        indexShader = it->second;

        res.target_code = m_shaders[indexShader];

        // Initialize with body resources always being required.
        // Mind that the zeroth resource is the invalid resource.
        for (mi::Size i = 1, n = res.target_code->get_texture_count(); i < n; ++i)
        {
            if (res.target_code->get_texture_is_body_resource(i))
            {
                res.textures.emplace_back(res.target_code->get_texture(i), res.target_code->get_texture_shape(i));
            }
        }

        for (mi::Size i = 1, n = res.target_code->get_light_profile_count(); i < n; ++i)
        {
            if (res.target_code->get_light_profile_is_body_resource(i))
            {
                res.light_profiles.emplace_back(res.target_code->get_light_profile(i));
            }
        }

        for (mi::Size i = 1, n = res.target_code->get_bsdf_measurement_count(); i < n; ++i)
        {
            if (res.target_code->get_bsdf_measurement_is_body_resource(i))
            {
                res.bsdf_measurements.emplace_back(res.target_code->get_bsdf_measurement(i));
            }
        }

        if (res.target_code->get_argument_block_count() > 0)
        {
            // Create argument block for the new compiled material and additional resources
            mi::base::Handle<Resource_callback> res_callback(new Resource_callback(transaction, res.target_code.get(), res));

            res.argument_block = res.target_code->create_argument_block(0, res.compiled_material.get(), res_callback.get());
        }
    }
    else
    {
        // Generate new target code.
        indexShader = static_cast<int>(m_shaders.size()); // The amount of different shaders in the code cache gives the next index.

        // Determine the material configuration by checking which minimal amount of expressions need to be generated as direct callable programs.
        ShaderConfiguration config;

        determineShaderConfiguration(res, config);

        // Build the required function descriptions for the expression required by the material configuration.
        std::vector<mi::neuraylib::Target_function_description> descs;

        const std::string suffix = std::to_string(indexShader);

        // These are all expressions required for a material which does everything supported in this renderer.
        // The Target_function_description only stores the C-pointers to the base names!
        // Make sure these are not destroyed as long as the descs vector is used.
        std::string name_init                           = "__direct_callable__init"                          + suffix;
        std::string name_thin_walled                    = "__direct_callable__thin_walled"                   + suffix;
        std::string name_surface_scattering             = "__direct_callable__surface_scattering"            + suffix;
        std::string name_surface_emission_emission      = "__direct_callable__surface_emission_emission"     + suffix;
        std::string name_surface_emission_intensity     = "__direct_callable__surface_emission_intensity"    + suffix;
        std::string name_surface_emission_mode          = "__direct_callable__surface_emission_mode"         + suffix;
        std::string name_backface_scattering            = "__direct_callable__backface_scattering"           + suffix;
        std::string name_backface_emission_emission     = "__direct_callable__backface_emission_emission"    + suffix;
        std::string name_backface_emission_intensity    = "__direct_callable__backface_emission_intensity"   + suffix;
        std::string name_backface_emission_mode         = "__direct_callable__backface_emission_mode"        + suffix;
        std::string name_ior                            = "__direct_callable__ior"                           + suffix;
        std::string name_volume_absorption_coefficient  = "__direct_callable__volume_absorption_coefficient" + suffix;
        std::string name_volume_scattering_coefficient  = "__direct_callable__volume_scattering_coefficient" + suffix;
        std::string name_volume_directional_bias        = "__direct_callable__volume_directional_bias"       + suffix;
        std::string name_geometry_cutout_opacity        = "__direct_callable__geometry_cutout_opacity"       + suffix;
        std::string name_hair_bsdf                      = "__direct_callable__hair"                          + suffix;

        // Centralize the init functions in a single material init().
        // This will only save time when there would have been multiple init functions inside the shader.
        // Also for very complicated materials with cutout opacity this is most likely a loss,
        // because the geometry.cutout is only needed inside the anyhit program and
        // that doesn't need additional evalations for the BSDFs, EDFs, or VDFs at that point.
        descs.push_back(mi::neuraylib::Target_function_description("init", name_init.c_str()));

        if (!config.is_thin_walled_constant)
        {
            descs.push_back(mi::neuraylib::Target_function_description("thin_walled", name_thin_walled.c_str()));
        }
        if (config.is_surface_bsdf_valid)
        {
            descs.push_back(mi::neuraylib::Target_function_description("surface.scattering", name_surface_scattering.c_str()));
        }
        if (config.is_surface_edf_valid)
        {
            descs.push_back(mi::neuraylib::Target_function_description("surface.emission.emission", name_surface_emission_emission.c_str()));
            if (!config.is_surface_intensity_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity", name_surface_emission_intensity.c_str()));
            }
            if (!config.is_surface_intensity_mode_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("surface.emission.mode", name_surface_emission_mode.c_str()));
            }
        }
        if (config.is_backface_bsdf_valid)
        {
            descs.push_back(mi::neuraylib::Target_function_description("backface.scattering", name_backface_scattering.c_str()));
        }
        if (config.is_backface_edf_valid)
        {
            if (config.use_backface_edf)
            {
                descs.push_back(mi::neuraylib::Target_function_description("backface.emission.emission", name_backface_emission_emission.c_str()));
            }
            if (config.use_backface_intensity && !config.is_backface_intensity_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("backface.emission.intensity", name_backface_emission_intensity.c_str()));
            }
            if (config.use_backface_intensity_mode && !config.is_backface_intensity_mode_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("backface.emission.mode", name_backface_emission_mode.c_str()));
            }
        }
        if (!config.is_ior_constant)
        {
            descs.push_back(mi::neuraylib::Target_function_description("ior", name_ior.c_str()));
        }
        if (!config.is_absorption_coefficient_constant)
        {
            descs.push_back(mi::neuraylib::Target_function_description("volume.absorption_coefficient", name_volume_absorption_coefficient.c_str()));
        }
        if (config.is_vdf_valid)
        {
            // The MDL SDK is NOT generating functions for VDFs! This would fail in ILink_unit::add_material().
            // descs.push_back(mi::neuraylib::Target_function_description("volume.scattering", name_volume_scattering.c_str()));

            // The scattering coefficient and directional bias are not used when there is no valid VDF.
            if (!config.is_scattering_coefficient_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("volume.scattering_coefficient", name_volume_scattering_coefficient.c_str()));
            }

            // FIXME This assumes the volume.scattering expression is exactly df::anisotropic_vdf(), not df::fog_vdf() or VDF mixers or modifiers.
            // config.is_directional_bias_constant == true for unsupported volume.scattering expressions and this description is not added.
            if (!config.is_directional_bias_constant)
            {
                descs.push_back(mi::neuraylib::Target_function_description("volume.scattering.directional_bias", name_volume_directional_bias.c_str()));
            }

            // volume.scattering.emission_intensity is not implemented.
        }

        // geometry.displacement is not implemented.

        // geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

        if (config.use_cutout_opacity)
        {
            descs.push_back(mi::neuraylib::Target_function_description("geometry.cutout_opacity", name_geometry_cutout_opacity.c_str()));
        }
        if (config.is_hair_bsdf_valid)
        {
            descs.push_back(mi::neuraylib::Target_function_description("hair", name_hair_bsdf.c_str()));
        }

        // Generate target code for the compiled material.
        mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(m_mdl_backend->create_link_unit(transaction, context.get()));

        mi::Sint32 reason = link_unit->add_material(res.compiled_material.get(), descs.data(), descs.size(), context.get());
        if (reason != 0)
        {
            std::cerr << "ERROR: compileMaterial() link_unit->add_material() returned " << reason << '\n';
        }
        if (!log_messages(context.get()))
        {
            std::cerr << "ERROR: compileMaterial() On link_unit->add_material()\n";
            return false;
        }

        res.target_code = mi::base::Handle<const mi::neuraylib::ITarget_code>(m_mdl_backend->translate_link_unit(link_unit.get(), context.get()));
        if (!log_messages(context.get()))
        {
            std::cerr << "ERROR: compileMaterial() On m_mdl_backend->translate_link_unit()\n";
            return false;
        }

        // Store the new shader index in the map.
        m_mapMaterialHashToShaderIndex[material_hash] = indexShader;

        // These two vectors are the same size:
        // Store the target code handle inside the shader cache.
        m_shaders.push_back(res.target_code);
        // Store the shader configuration to be able to build the required direct callables on the device later.
        m_shaderConfigurations.push_back(config);

        // Add all used resources. The zeroth entry is the invalid resource.
        for (mi::Size i = 1, n = res.target_code->get_texture_count(); i < n; ++i)
        {
            res.textures.emplace_back(res.target_code->get_texture(i), res.target_code->get_texture_shape(i));
        }

        if (res.target_code->get_light_profile_count() > 0)
        {
            for (mi::Size i = 1, n = res.target_code->get_light_profile_count(); i < n; ++i)
            {
                res.light_profiles.emplace_back(res.target_code->get_light_profile(i));
            }
        }

        if (res.target_code->get_bsdf_measurement_count() > 0)
        {
            for (mi::Size i = 1, n = res.target_code->get_bsdf_measurement_count(); i < n; ++i)
            {
                res.bsdf_measurements.emplace_back(res.target_code->get_bsdf_measurement(i));
            }
        }

        if (res.target_code->get_argument_block_count() > 0)
        {
            res.argument_block = res.target_code->get_argument_block(0);
        }

#if 0 // DEBUG Print or write the PTX code when a new shader has been generated.
    if (res.target_code)
    {
      std::string code = res.target_code->get_code();

      // Print generated PTX source code to the console.
      //std::cout << code << std::endl;

      // Dump generated PTX source code to a local folder for offline comparisons.
      const std::string filename = std::string("./mdl_ptx/") + material_simple_name + std::string("_") + getDateTime() + std::string(".ptx");

      saveString(filename, code);
    }
#endif // DEBUG

    } // End of generating new target code.

    // Build the material information for this material reference.
    mi::base::Handle<mi::neuraylib::ITarget_value_layout const> arg_layout;

    if (res.target_code->get_argument_block_count() > 0)
    {
        arg_layout = res.target_code->get_argument_block_layout(0);
    }

    // Store the material information per reference inside the MaterialMDL structure.
    materialMDL->storeMaterialInfo(indexShader,
                                   material_definition.get(),
                                   res.compiled_material.get(),
                                   arg_layout.get(),
                                   res.argument_block.get());

    // Now that the code and the resources are setup as MDL handles,
    // call into the Device class to setup the CUDA and OptiX resources.

    return true;
}


bool MdlWrapper::isEmissiveShader(const int indexShader) const
{
    bool result = false;

    if (0 <= indexShader && indexShader < m_shaderConfigurations.size())
    {
        result = m_shaderConfigurations[indexShader].isEmissive();
    }
    else
    {
        std::cout << "ERROR: isEmissiveShader() called with invalid index " << indexShader << '\n';
    }

    return result;
}
