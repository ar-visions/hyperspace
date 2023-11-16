#pragma once
#include <media/image.hpp>
#include <async/async.hpp>
namespace ion {

enums(VideoFormat, undefined,
    undefined, YUY2, NV12, MJPEG, H264);

async camera(array<VideoFormat> priority, str alias, int rwidth, int rheight, lambda<void(image& img)> frame);

}