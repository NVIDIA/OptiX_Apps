#ifndef RENDERINGGUI_H
#define RENDERINGGUI_H

#include "pane_flags.h"
#include <stdint.h>

struct RenderingGUI {
    int32_t num_panes;

    PaneFlags pane_a;
    PaneFlags pane_b;
    PaneFlags pane_c;
};


#endif // RENDERINGGUI_H
