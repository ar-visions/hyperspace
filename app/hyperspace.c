// ------------------------------------------------------
// hyperspace
// ------------------------------------------------------
//      1. record:
//           save images with json:
//               plot head-pos/look-offset on all cameras
//               plot hand motions with gestures from-to (this is how we get accurate gestures in time)
//      2. annotate:
//           circle head scaling (orientation irrelevant)
//           plotting for eyes
//      3. train:
//           needs ability to select models from checkpoints (likely a list-view for this data)
//           no ability to select until a threshold met; 10,000 data iterations, potentially
//      4. testbed with controls
//           button, sliders, etc that are hyperspace controlled
// ------------------------------------------------------
// capture look  | - target for head and look offset, very simple sequence (less than 120 lines of code!)
//               |
// capture point | - point-to target as it moves, we dont need to record long and the user can change their way of pointing many times
//               |
// annotate      | - train data for basic targeting and sizing of features (hands, face)
//               |
// train         | - list-view; perhaps a visualization of its training curve, a selection in the graph point showing the data below
//               |
// explore       | - use the model, with several test areas
//               |
// ------------------------------------------------------
#include <import>
#include <math.h>

object hyperspace_window_mouse(hyperspace a, event e) {
    return null;
}

object hyperspace_background(hyperspace a, object arg) {
    return null;
}

map    hyperspace_interface(hyperspace a, object arg) {
    map m = map_of(
        "main", pane(),
        null);
    return m;
}

// should not need any background model for target to render its clear color
// test canvas painting after it draws clear-color
// issue now is pipeline is not 
none hyperspace_init(hyperspace a) {
    trinity t   = a->t = trinity();
    window  win = a->w = window(
        t, t, title, string("hyperspace"),
        width, 400, height, 400); // needs a fullscreen method
    vec4f   bg  = vec4f(0.22f, 0.22f, 0.22f, 1.0f);
    a->r_background  = target (w, win, id, 22, wscale, 1.0f, clear_color, bg);
    initialize(a);
}

none hyperspace_dealloc(hyperspace a) { }

int main(int argc, cstrs argv) {
    hyperspace a = hold(hyperspace(argv));
    return run(a);
}

define_class(hyperspace, app)