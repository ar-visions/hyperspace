#include <core/core.hpp>
#include <async/async.hpp>
#include <net/net.hpp>
#include <math/math.hpp>
#include <media/media.hpp>

using namespace ion;

/// ------------------------------------------------------------
/// audio annotation app first
/// ------------------------------------------------------------

int main() {
    str  api_key     = "AIzaSyAg4nh93xKESkGZvv7Ocv2PBBFAM1jyDSs";
    str  channel_id  = "UCpVm7bg6pXKo1Pr6k5kxG9A";
    uri  request_url = fmt { "https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={0}&maxResults=1&key={1}", { channel_id, api_key }};
    
    /// switch to vulkan display, get working now.  json from https works
    request(request_url, null).then([](mx res) {
        message msg { res.grab() };
        var content = msg->content;
        console.log("item 0 etag: {0}", { content["items"][0]["etag"] });
    }).except([](mx err) {
        console.error("http request error: {0}", { err });
    });
    ///
    return async::await();
}
