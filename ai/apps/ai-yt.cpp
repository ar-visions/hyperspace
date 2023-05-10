#include <core/core.hpp>
#include <async/async.hpp>
#include <net/net.hpp>

using namespace ion;

/// could train lots of things just from national geographic-like videos
/// how animals move in space relative to other animals.
/// thats interesting because you can certainly map lots of information
/// ----------------------------
int main() {
    lambda<memory*(memory*)> mtest;
    int test = 1;
    ///
    str  api_key     = "AIzaSyAg4nh93xKESkGZvv7Ocv2PBBFAM1jyDSs";
    str  channel_id  = "UCpVm7bg6pXKo1Pr6k5kxG9A";
    uri  request_url = fmt { "https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={0}&maxResults=1&key={1}", { channel_id, api_key }};
    ///
    request(request_url, null).then([](mx result) {
        message::members &rs_0 = result.ref<message::members>();
        //struct members {
        //    uri     query;
        //    mx      code = int(0);
        //    map<mx> headers;
        //    mx      content; /// best to store as mx, so we can convert directly in lambda arg, functionally its nice to have delim access in var.
        //} &m;
        console.log("code: {0}", {rs_0.code, rs_0.headers});
        printf("abc\n");
    });
    
    ///
    return async::await();
}
