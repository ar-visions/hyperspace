#include <core/core.hpp>
#include <async/async.hpp>
#include <net/net.hpp>
#include <math/math.hpp>
#include <image/image.hpp>

using namespace ion;
/// could train lots of things just from national geographic-like videos
/// how animals move in space relative to other animals.
/// thats interesting because you can certainly map lots of information
/// ----------------------------
int main() {
    str  api_key     = "AIzaSyAg4nh93xKESkGZvv7Ocv2PBBFAM1jyDSs";
    str  channel_id  = "UCpVm7bg6pXKo1Pr6k5kxG9A";
    uri  request_url = fmt { "https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={0}&maxResults=1&key={1}", { channel_id, api_key }};
    ///
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
