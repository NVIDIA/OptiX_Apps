#ifndef PANE_FLAGS_H
#define PANE_FLAGS_H

struct PaneFlags {
    bool do_reference;
    bool do_ris;
    bool do_spatial_reuse;
    bool do_temporal_reuse;
    bool do_nrc;
};

static constexpr PaneFlags ref_pane_flags = {true, false, false, false, false};

#endif // PANE_FLAGS_H
