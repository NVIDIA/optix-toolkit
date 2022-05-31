//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <sutil/vec_math.h>


namespace sutil {

// implementing a perspective camera
class Camera {
public:
    Camera()
        : m_eye(make_float3(1.0f)), m_lookat(make_float3(0.0f)), m_up(make_float3(0.0f, 1.0f, 0.0f)), m_fovY(35.0f), m_aspectRatio(1.0f)
    {
    }

    Camera(const float3& eye, const float3& lookat, const float3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    float3 direction() const { return normalize(m_lookat - m_eye); }
    void setDirection(const float3& dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

    const float3& eye() const { return m_eye; }
    void setEye(const float3& val) { m_eye = val; }
    const float3& lookat() const { return m_lookat; }
    void setLookat(const float3& val) { m_lookat = val; }
    const float3& up() const { return m_up; }
    void setUp(const float3& val) { m_up = val; }
    const float& fovY() const { return m_fovY; }
    void setFovY(const float& val) { m_fovY = val; }
    const float& aspectRatio() const { return m_aspectRatio; }
    void setAspectRatio(const float& val) { m_aspectRatio = val; }

    // UVW forms an orthogonal, but not orthonormal basis!
    void UVWFrame(float3& U, float3& V, float3& W) const;

private:
    float3 m_eye;
    float3 m_lookat;
    float3 m_up;
    float m_fovY;
    float m_aspectRatio;
};

}
