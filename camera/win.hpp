#pragma once
#include <media/image.hpp>
#include <async/async.hpp>
namespace ion {
async win_capture(lambda<bool(image& img)> frame);
}