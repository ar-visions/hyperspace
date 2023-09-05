#include <mx/mx.hpp>
#include <async/async.hpp>
#include <net/net.hpp>
#include <math/math.hpp>
#include <media/media.hpp>
#include <ux/ux.hpp>

using namespace ion;

/// ------------------------------------------------------------
/// audio annotation app first
/// ------------------------------------------------------------

struct annotate:Element {
    struct props {
        int sample;
        callback handler;
        ///
        doubly<prop> meta() {
            return {
                prop { "sample",  sample },
                prop { "handler", handler}
            };
        }
    };

    component(annotate, Element, props);

    void mounting() {
        console.log("mounting");
    }

    Element render() {
        return button {
            { "content", fmt {"hello world: {0}", { state->sample }} },
            { "on-click",
                callback([&](event e) {
                    console.log("on-click...");
                    if (state->handler)
                        state->handler(e);
                })
            }
        };
    }
};

int main() {
    return app([](app &ctx) -> Element {
        return annotate {
            { "id",     "main"  }, /// id should be a name of component if not there
            { "sample",  int(2) },
            { "on-silly",
                callback([](event e) {
                    console.log("on-silly");
                })
            }
        };
    });
}
