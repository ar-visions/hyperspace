#include <core/core.hpp>
#include <async/async.hpp>
#include <net/net.hpp>
#include <math/math.hpp>
#include <media/media.hpp>
#include <ux/ux.hpp>

using namespace ion;

/// ------------------------------------------------------------
/// audio annotation app first
/// ------------------------------------------------------------

struct annotate:node {
    struct props {
        int sample;
        callback handler;
    } &m;
    
    ///
    ctr_args(annotate, node, props, m);
    
    ///
    doubly<prop> meta() {
        return {
            prop { m, "sample",  m.sample },
            prop { m, "handler", m.handler}
        };
    }
    
    ///
    void mounting() {
        console.log("mounting");
    }

    /// if no render is defined, the content is used for embedding children from content (if its there)
    /// if there is a render the content can be used within it
    Element render() {
        return button {
            { "content", fmt {"hello world: {0}", { m.sample }} },
            { "on-click",
                callback([&](event e) {
                    console.log("on-click...");
                    if (m.handler)
                        m.handler(e);
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
