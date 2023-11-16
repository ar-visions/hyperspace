
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

void yuy2_rgba(const uint8_t* yuy2, uint8_t* rgba, int width, int height) {
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

void nv12_rgba(const uint8_t* nv12, uint8_t* rgba, int width, int height) {
    const uint8_t* Y = nv12;
    const uint8_t* UV = nv12 + (width * height);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int Yindex = y * width + x;
            int UVindex = (y / 2) * width + (x & ~1);

            // Y component
            int Yvalue = Y[Yindex] - 16;

            // UV components
            int U = UV[UVindex] - 128;
            int V = UV[UVindex + 1] - 128;

            // Set RGBA values
            rgba8 *dst = (rgba8*)(rgba + 4 * Yindex);
            dst->r = clamp((298 * Yvalue + 409 * V + 128) >> 8);
            dst->g = clamp((298 * Yvalue - 100 * U - 208 * V + 128) >> 8);
            dst->b = clamp((298 * Yvalue + 516 * U + 128) >> 8);
            dst->a = 255;
        }
}

VideoFormat video_format_from_guid(GUID &subtype) {
    if (subtype == MFVideoFormat_MJPG)
        return VideoFormat::MJPEG;
    if (subtype == MFVideoFormat_YUY2)
        return VideoFormat::YUY2;
    if (subtype == MFVideoFormat_NV12)
        return VideoFormat::NV12;
    if (subtype == MFVideoFormat_H264)
        return VideoFormat::H264;
    return VideoFormat::undefined;
}

// Custom hash function for GUID
struct GuidHash {
    size_t operator()(const GUID& guid) const {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(&guid);
        std::hash<uint64_t> hash;
        return hash(p[0]) ^ hash(p[1]);
    }
};

// Custom equality function for GUID
struct GuidEqual {
    bool operator()(const GUID& lhs, const GUID& rhs) const {
        return (memcmp(&lhs, &rhs, sizeof(GUID)) == 0);
    }
};

/// call stop on the sync(true) to force stop the service, and return its mx result
async camera(array<VideoFormat> priority, str alias, int rwidth, int rheight, lambda<void(image& img)> frame) {
    
    /// start capture service
    return async(1, [priority, alias, frame, rwidth, rheight](runtime *rt, int i) -> mx {
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        assert(!FAILED(hr));

        // Initialize Media Foundation
        MFStartup(MF_VERSION);

        IMFMediaSource* pSource     = null;
        IMFAttributes*  pConfig     = null;
        IMFActivate**   ppDevices   = null;
        UINT32          deviceCount = 0;
        VideoFormat     selected_format;
        int             format_index    = -1;
        int             selected_stream = -1;

        // enum video capture devices, select one we have highest priority for
        MFCreateAttributes(&pConfig, 1);
        pConfig->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        MFEnumDeviceSources(pConfig, &ppDevices, &deviceCount);
        IMFMediaType* pType = null;

        /// using while so we can break early (format not found), or at end of statement
        for (int i = 0; i < deviceCount; i++) {
            u32 ulen = 0;
            
            ppDevices[i]->GetStringLength(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &ulen);
            utf16 device_name { size_t(ulen + 1) };
            ppDevices[i]->GetString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                (WCHAR*)device_name.data, ulen + 1, &ulen);
            cstr utf8_name  = device_name.to_utf8();
            bool name_match = true;
            if (alias && !strstr(utf8_name, alias.data))
                name_match = false;
            
            free(utf8_name);
            if (!name_match)
                continue;
            ppDevices[i]->ActivateObject(IID_PPV_ARGS(&pSource));

            /// find the index of media source we want to select
            DWORD stream_index   = 0;
            GUID  subtype        = GUID_NULL;
            IMFSourceReader* pReader = null;
            MFCreateSourceReaderFromMediaSource(pSource, null, &pReader);
            while (SUCCEEDED(pReader->GetNativeMediaType(stream_index, 0, &pType))) {
                pType->GetGUID(MF_MT_SUBTYPE, &subtype);
                VideoFormat vf = video_format_from_guid(subtype);
                if (vf != VideoFormat::undefined) {
                    int index = priority.index_of(vf);
                    if (index >= 0 && (index < format_index || format_index == -1)) {
                        selected_format = vf;
                        format_index    = index;
                        selected_stream = stream_index;
                    }
                }
                pType->Release();
                pType = null;
                stream_index++;
            }
            
            if (!selected_format)
                continue;
            
            /// load selected_stream (index of MediaFoundation device stream)
            if (SUCCEEDED(pReader->GetNativeMediaType(selected_stream, 0, &pType))) {
                // set resolution; asset success
                HRESULT hr = MFSetAttributeSize(pType, MF_MT_FRAME_SIZE, rwidth, rheight);
                assert(SUCCEEDED(hr));
                pReader->SetCurrentMediaType(selected_stream, null, pType);
            }

            if (pType) {
                UINT32 width = 0, height = 0;

                /// get frame size; verify its the same as requested after we set it
                assert(SUCCEEDED(MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height)));
                assert(rwidth == width && rheight == height);
                wprintf(L"width: %u, height: %u\n", width, height);

                rgba8 *px  = (rgba8*)calloc(sizeof(rgba8), width * height);
                image  img = image(size { height, width }, (rgba8*)px, 0);

                /// capture loop
                while (!rt->stop) {
                    DWORD      flags;
                    IMFSample* pSample = null;
                    pReader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, null, &flags, null, &pSample);
                    if (flags & MF_SOURCE_READERF_ENDOFSTREAM) /// end of stream
                        break;

                    if (pSample) {
                        IMFMediaBuffer* pBuffer         = null;
                        BYTE*           pData           = null;
                        DWORD           maxLength       = 0, 
                                        currentLength   = 0;

                        pSample->ConvertToContiguousBuffer(&pBuffer);
                        pBuffer->Lock(&pData, &maxLength, &currentLength);

                        switch (selected_format.value) {
                            case VideoFormat::YUY2:
                                // each pixel in YUY2 is 2 bytes: Y0 U0 Y1 V0, Y2 U1 Y3 V1, ...
                                yuy2_rgba(pData, (BYTE*)img.data, width, height);
                                break;
                            case VideoFormat::NV12:
                                nv12_rgba(pData, (BYTE*)img.data, width, height);
                                break;
                            case VideoFormat::MJPEG:
                                assert(false);
                                break;
                            default:
                                assert(false);
                                break; 
                        }
                        pBuffer->Unlock();
                        pBuffer->Release();
                        pSample->Release();

                        /// call user function -- dont call sync(true) from this procedure, for obvious raisonettes
                        frame(img);
                    }
                }
                /// cleanup
                pType->Release();
            }
            pReader->Release();
            pSource->Release();
            ppDevices[0]->Release();
            CoTaskMemFree(ppDevices);
            break;
        }

        if (!selected_format)
            console.log("camera: no suitable device/format found");
        
        pConfig->Release();
        MFShutdown();
        return pType != null;
    });
}
}
