__kernel void update_fields(__global float* temperature,
                            __global float* pressure,
                            int nx, int ny, int nz) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int idx = x + y*nx + z*nx*ny;
    if (x >= nx || y >= ny || z >= nz) return;

    float t = temperature[idx];
    float p = pressure[idx];

    temperature[idx] = t + 0.01f * (p - 100000.0f) * dt;
    pressure[idx] = p - 0.02f * (t - 290.0f) * dt;
}
