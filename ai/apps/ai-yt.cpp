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
    
    ///
    mtest = lambda<memory*(memory*)>([](memory* in) -> memory* {
        return in;
    });
    
    ///
    lambda<mx(size_t)> fn = [](size_t s) -> mx {
        return int(5) + int(s);
    };
    
    ///
    i32  abc         = i32(fn(1));
    bool fn_test     = !fn;
    str  api_key     = "AIzaSyAg4nh93xKESkGZvv7Ocv2PBBFAM1jyDSs";
    str  channel_id  = "UCpVm7bg6pXKo1Pr6k5kxG9A";
    uri  request_url = fmt { "https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={0}&maxResults=1&key={1}", { channel_id, api_key }};
    
    console.log("url = {0}", { request_url });
    
    ///
    request(request_url, null).then([](mx result) {
        console.log("got a result");
    });
    
    ///
    return async::await();
}
