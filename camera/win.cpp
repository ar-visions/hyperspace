
#include <camera/win.hpp>

//#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wmcodecdsp.h>
#include <wchar.h>

namespace ion {

int clamp(int v) {
    return v < 0 ? 0 : v > 255 ? 255 : v;
}

void yuy2_rgba(const BYTE* yuy2, BYTE* rgba, int width, int height) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x += 2) {
            // Extract YUY2 components
            uint8_t Y0 = yuy2[0];
            uint8_t U  = yuy2[1];
            uint8_t Y1 = yuy2[2];
            uint8_t V  = yuy2[3];

            // Convert YUV to RGB
            int C = Y0 - 16;
            int D = U - 128;
            int E = V - 128;

            rgba[0] = (uint8_t)clamp(( 298 * C + 409 * E + 128) >> 8); // R0
            rgba[1] = (uint8_t)clamp(( 298 * C - 100 * D - 208 * E + 128) >> 8); // G0
            rgba[2] = (uint8_t)clamp(( 298 * C + 516 * D + 128) >> 8); // B0
            rgba[3] = 255; // A0

            C = Y1 - 16;

            rgba[4] = (uint8_t)clamp(( 298 * C + 409 * E + 128) >> 8); // R1
            rgba[5] = (uint8_t)clamp(( 298 * C - 100 * D - 208 * E + 128) >> 8); // G1
            rgba[6] = (uint8_t)clamp(( 298 * C + 516 * D + 128) >> 8); // B1
            rgba[7] = 255; // A1

            yuy2 += 4;
            rgba += 8;
        }
}

async win_capture(lambda<bool(image& img)> frame) {
    
    /// start capture service
    return async(1, [frame](runtime *rt, int i) -> mx {
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        if (FAILED(hr)) {
            // Handle error
        }
        // Initialize Media Foundation
        MFStartup(MF_VERSION);

        IMFMediaSource* pSource     = null;
        IMFAttributes*  pConfig     = null;
        IMFActivate**   ppDevices   = null;
        UINT32          deviceCount = 0;

        // Enumerate video capture devices
        MFCreateAttributes(&pConfig, 1);
        pConfig->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        MFEnumDeviceSources(pConfig, &ppDevices, &deviceCount);

        if (deviceCount == 0)
            return false;

        ppDevices[0]->ActivateObject(IID_PPV_ARGS(&pSource));

        IMFMediaType* pType = null;
        IMFSourceReader* pReader = null;
        MFCreateSourceReaderFromMediaSource(pSource, null, &pReader);

        DWORD streamIndex = 0;
        GUID  subtype = GUID_NULL;
        while (SUCCEEDED(pReader->GetNativeMediaType(streamIndex, 0, &pType))) {
            pType->GetGUID(MF_MT_SUBTYPE, &subtype);

            WCHAR subtypeName[40]; // Allocate a buffer for the string
            StringFromGUID2(subtype, subtypeName, ARRAYSIZE(subtypeName));
            wprintf(L"Subtype: %s\n", subtypeName);

            // Set the resolution to 1920x1080, or 640x360 (16:9 formats; useful in arg form)
            if (subtype == MFVideoFormat_MJPG || subtype == MFVideoFormat_YUY2) {
                HRESULT hr = MFSetAttributeSize(pType, MF_MT_FRAME_SIZE, 1920, 1080);
                if (!SUCCEEDED(hr))
                    hr = MFSetAttributeSize(pType, MF_MT_FRAME_SIZE, 640, 360);
                assert(SUCCEEDED(hr));
                pReader->SetCurrentMediaType(streamIndex, null, pType);
                break;
            }

            pType->Release();
            pType = null;
            streamIndex++;
        }

        if (!pType)
            return false;

        UINT32 width = 0, height = 0;
        // Extract the width and height
        if (SUCCEEDED(MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height)))
            wprintf(L"Width: %u, Height: %u\n", width, height);

        rgba8 *px  = (rgba8*)calloc(sizeof(rgba8), width * height);
        image  img = image(size { height, width }, (rgba8*)px, 0);

        // Capture loop
        for (;;) {
            DWORD flags;
            IMFSample* pSample = null;
            pReader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, null, &flags, null, &pSample);

            if (flags & MF_SOURCE_READERF_ENDOFSTREAM)
                break; // end of stream

            if (pSample) {
                IMFMediaBuffer* pBuffer = null;
                BYTE* pData = null;
                DWORD maxLength = 0, currentLength = 0;

                pSample->ConvertToContiguousBuffer(&pBuffer);
                pBuffer->Lock(&pData, &maxLength, &currentLength);

                // Now pData points to the YUY2 data
                // Each pixel in YUY2 is 2 bytes: Y0 U0 Y1 V0, Y2 U1 Y3 V1, ...
                if (subtype == MFVideoFormat_YUY2)
                    yuy2_rgba(pData, (BYTE*)img.data, width, height);
                else if (subtype == MFVideoFormat_MJPG)
                    assert(false);
                
                pBuffer->Unlock();
                pBuffer->Release();
                pSample->Release();

                /// call user function
                if (!frame(img))
                    break;
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
        return true;
    });
}
}
