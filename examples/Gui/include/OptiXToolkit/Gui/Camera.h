// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/vec_math.h>


namespace otk {

// implementing a perspective camera
class Camera {
public:
    Camera()
        : m_eye(make_float3(1.0f)), m_lookAt(make_float3(0.0f)), m_up(make_float3(0.0f, 1.0f, 0.0f)), m_fovY(35.0f), m_aspectRatio(1.0f)
    {
    }

    Camera(const float3& eye, const float3& lookAt, const float3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookAt(lookAt), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    float3 direction() const { return normalize(m_lookAt - m_eye); }
    void setDirection(const float3& dir) { m_lookAt = m_eye + length(m_lookAt - m_eye) * dir; }

    const float3& eye() const { return m_eye; }
    void setEye(const float3& val) { m_eye = val; }
    const float3& lookAt() const { return m_lookAt; }
    void setLookAt(const float3& val) { m_lookAt = val; }
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
    float3 m_lookAt;
    float3 m_up;
    float m_fovY;
    float m_aspectRatio;
};

} // namespace otk
