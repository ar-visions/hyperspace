
#include <stdio.h>

#if defined(__APPLE__)

#include <mx/mx.hpp>
#include <camera/camera.hpp>

using namespace ion;

void on_data(void *user, void *data, int len) {
    printf("camera-test: %d\n", len);
}

int main(int argc, char* argv[]) {
    printf("stupid\n");
    usleep(100000);
    Camera cam(null, on_data);
    printf("starting capture\n");
    cam.start_capture();

    while (1) {
        usleep(10000);
    }
}

#elif defined(__linux__)

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
  // Open the video device file
  int fd = open("/dev/video0", O_RDONLY);
  if (fd < 0) {
    perror("open");
    exit(1);
  }

  // Set the video capture format to MJPEG
  struct v4l2_format fmt;
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;

  int ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
  if (ret < 0) {
    perror("ioctl");
    exit(1);
  }

  // Read an MJPEG frame
  unsigned char buffer[fmt.fmt.pix.sizeimage];
  ret = read(fd, buffer, fmt.fmt.pix.sizeimage);
  if (ret < 0) {
    perror("read");
    exit(1);
  }

  // Process the MJPEG frame
  // TODO: Process the MJPEG frame here

  // Close the video device file
  close(fd);

  return 0;
}

#else

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wmcodecdsp.h>

int main() {
// Initialize Media Foundation
    MFStartup(MF_VERSION);

    IMFMediaSource* pSource = nullptr;
    IMFAttributes* pConfig = nullptr;
    IMFActivate** ppDevices = nullptr;
    UINT32 deviceCount = 0;

    // Enumerate video capture devices
    MFCreateAttributes(&pConfig, 1);
    pConfig->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    MFEnumDeviceSources(pConfig, &ppDevices, &deviceCount);

    if (deviceCount == 0) {
        // No devices found
        return 1;
    }

    ppDevices[0]->ActivateObject(IID_PPV_ARGS(&pSource));

    IMFMediaType* pType = nullptr;
    IMFSourceReader* pReader = nullptr;
    MFCreateSourceReaderFromMediaSource(pSource, NULL, &pReader);

    DWORD streamIndex = 0;
    while (SUCCEEDED(pReader->GetNativeMediaType(streamIndex, 0, &pType))) {
        GUID subtype = GUID_NULL;
        pType->GetGUID(MF_MT_SUBTYPE, &subtype);

        WCHAR subtypeName[40]; // Allocate a buffer for the string
        StringFromGUID2(subtype, subtypeName, ARRAYSIZE(subtypeName));
        wprintf(L"Subtype: %s\n", subtypeName);

        if (subtype == MFVideoFormat_H264) {
            pReader->SetCurrentMediaType(streamIndex, NULL, pType);
            break;
        }
        pType->Release();
        pType = nullptr;
        streamIndex++;
    }

    if (!pType) {
        // H.264 format not found
        return 2;
    }

    // Capture loop
    while (true) {
        DWORD flags;
        IMFSample* pSample = nullptr;
        pReader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, nullptr, &flags, nullptr, &pSample);

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            break; // end of stream
        }

        if (pSample) {
            // Process the H.264 sample
            // ... 

            pSample->Release();
        }
    }

    // Cleanup
    pType->Release();
    pReader->Release();
    pSource->Release();
    ppDevices[0]->Release();
    CoTaskMemFree(ppDevices);
    pConfig->Release();

    MFShutdown();
    return 0;
}

#endif