#ifndef MDL_WRAPPER_H
#define MDL_WRAPPER_H

#include "inc/MaterialMDL.h"

#include "inc/CompileResult.h"
#include "inc/ShaderConfiguration.h"
#include "inc/Device.h"

#include <mi/mdl_sdk.h>
#include <mi/base/config.h>


class MdlWrapper {
public:
    bool initMDL(const std::vector<std::string>& searchPaths);

    mi::neuraylib::INeuray* load_and_get_ineuray(const char* filename);
    mi::Sint32 load_plugin(mi::neuraylib::INeuray* neuray, const char* path);
    bool log_messages(mi::neuraylib::IMdl_execution_context* context);
    void determineShaderConfiguration(const Compile_result& res, ShaderConfiguration& config);

    bool compileMaterial(mi::neuraylib::ITransaction* transaction, MaterialMDL* materialMDL, Compile_result& res);

    void initMaterialsMDL(std::vector<MaterialMDL*>& materialsMDL, std::vector<Device*>& devices_active);
    void initMaterialMDL(MaterialMDL* materialMDL, std::vector<Device*>& devices_active);

    bool isEmissiveShader(const int indexShader) const;

    void shutdownMDL();
public:

    mi::base::Handle<mi::base::ILogger> m_logger;

    // The last error message from MDL SDK.
    std::string m_last_mdl_error;

    mi::base::Handle<mi::neuraylib::INeuray>                m_neuray;
    mi::base::Handle<mi::neuraylib::IMdl_compiler>          m_mdl_compiler;
    mi::base::Handle<mi::neuraylib::ILogging_configuration> m_logging_config;
    mi::base::Handle<mi::neuraylib::IMdl_configuration>     m_mdl_config;
    mi::base::Handle<mi::neuraylib::IDatabase>              m_database;
    mi::base::Handle<mi::neuraylib::IScope>                 m_global_scope;
    mi::base::Handle<mi::neuraylib::IMdl_factory>           m_mdl_factory;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_execution_context;
    mi::base::Handle<mi::neuraylib::IMdl_backend>           m_mdl_backend;
    mi::base::Handle<mi::neuraylib::IImage_api>             m_image_api;

    // Maps a compiled material hash to a shader code cache index == shader configuration index.
    std::map<mi::base::Uuid, int> m_mapMaterialHashToShaderIndex;
    // These two vectors have the same size and implement shader reuse (references with the same MDL material).
    std::vector<mi::base::Handle<mi::neuraylib::ITarget_code const>> m_shaders;
    std::vector<ShaderConfiguration>                                 m_shaderConfigurations;

    // MDL specific things.

#ifdef MI_PLATFORM_WINDOWS
    HMODULE m_dso_handle = 0;
#else
    void* m_dso_handle = 0;
#endif

};

#endif // MDL_WRAPPER_H
