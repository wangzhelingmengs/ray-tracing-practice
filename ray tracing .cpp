
#include "stb_image.h"
#include "geometry.h"

#include <fstream>
#include <vector>
#include <limits>

#include<iostream>
#include<algorithm>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\types_c.h>
using namespace std;
using namespace cv;
Mat img, new_img1, new_img2;	   //img:原始图像; new_img:木质贴图，new_img2地球贴图;
int Rr[1000][1000], Gg[1000][1000], Bb[1000][1000];
int Rr1[1000][1000], Gg1[1000][1000], Bb1[1000][1000];//贴图1的RGB信息
int Rr2[1000][1000], Gg2[1000][1000], Bb2[1000][1000];//贴图2的RGB信息
void read_rgb()//交换红蓝
{
    for (int row = 0; row < new_img1.rows; row++)
    {
        for (int col = 0; col < new_img1.cols; col++)
        {      
            Rr1[row][col] = new_img1.at<Vec3b>(row, col)[2];
            Gg1[row][col]= new_img1.at<Vec3b>(row, col)[1];
            Bb1[row][col]=new_img1.at<Vec3b>(row, col)[0];
        }
    }
    for (int row = 0; row < new_img2.rows; row++)
    {
        for (int col = 0; col < new_img2.cols; col++)
        { 
            Rr2[row][col] = new_img2.at<Vec3b>(row, col)[2];
            Gg2[row][col] = new_img2.at<Vec3b>(row, col)[1];
            Bb2[row][col] = new_img2.at<Vec3b>(row, col)[0];
        }
    }
}
//光源
struct Light
{
    Light(const vec3 &p, const float &i) : position(p), intensity(i) {}
    vec3 position;//发光点位置
    float intensity;//发光强度
};
//材质
struct Material
{
    
    Material(const float &r, const vec4 &a, const vec3 &color, const float &spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(vec4{1, 0, 0, 0}), diffuse_color(vec3{0, 0, 0}), specular_exponent(0) {}
    float refractive_index;//折射率
    vec4 albedo;//光的各部分权重,漫反射，高光，反射，折射
    vec3 diffuse_color;//材料本色
    float specular_exponent;//高光系数
};

struct Sphere
{
    vec3 center;//球心
    float radius;//球的半径
    Material material;

    Sphere(const vec3 &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}
    //判断射线和这个球是否相交，相交返回true
    bool ray_intersect(const vec3 &orig, const vec3 &dir, float &t0) const
    {
        vec3 L = center - orig;
        float tca = L * dir;
        float d2 = L * L - tca * tca;
        if (d2 > radius * radius)
            return false;
        float thc = sqrtf(radius * radius - d2);
        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0)
            t0 = t1;
        //cout << t0 << endl;
        if (t0 < 0)
            return false;
        return true;
    }
};
struct Box
{
    Box(const vec3& vmin, const vec3& vmax, const Material &m)
    {
        bounds[0] = vmin;
        bounds[1] = vmax;
        material = m;
        
    }
    vec3 bounds[2];
    Material material;
    
    //判断射线和这个长方体是否相交，相交返回true
    bool ray_intersect(const vec3& orig, const vec3& dir, float& t0) const
    {
        vec3 bmin, bmax;
        bmin = bounds[0];
        bmax = bounds[1];
        float tmin = (bmin.x - orig.x) / dir.x;
        float tmax = (bmax.x - orig.x) / dir.x;

        if (tmin > tmax) { swap(tmin, tmax); }

        float tymin = (bmin.y - orig.y) / dir.y;
        float tymax = (bmax.y - orig.y) / dir.y;

        if (tymin > tymax) { swap(tymin, tymax); }

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin) {

            tmin = tymin;
       
        }
        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (bmin.z - orig.z) / dir.z;
        float tzmax = (bmax.z - orig.z) / dir.z;
      
        if (tzmin > tzmax) { swap(tzmin, tzmax);  }

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        if (tzmin > tmin) {
            tmin = tzmin;
            
        }
            
        if (tzmax < tmax)
            tmax = tzmax;
        t0 = tmin;
        //cout << t0 << endl;
        return true;
    }
};
//反射，入射方向，和法线方向，返回出射方向
vec3 reflect(const vec3 &I, const vec3 &N)
{
    return I - N * 2.f * (I * N);
    
}
//折射，入射，法线方向，折射率，返回折射方向
vec3 refract(const vec3 &I, const vec3 &N, const float &refractive_index)
{ // Snell's law
    float cosi = -std::max(-1.f, std::min(1.f, I * N));
    float etai = 1, etat = refractive_index;
    vec3 n = N;
    if (cosi < 0)
    { // if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi;
        std::swap(etai, etat);
        n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? vec3{0, 0, 0} : I * eta + n * (eta * cosi - sqrtf(k));
}
float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//映射关系，输入交点，圆心坐标，贴图号，返回球面坐标对应的二维坐标对应的颜色值
vec3 maps_picture(vec3 point,vec3 center,int k) {
    float r = 0.0, g = 0.0, b = 0.0, u, v,pi=3.1415926;
    int nx, ny;
    nx = 960;
    ny = 600;
//空间坐标映射到球坐标
    float X, Y, Z;
    Y = point.y - center.y;
    Z = point.z - center.z;
    X = point.x - center.x;
    float sum = sqrt(X * X + Y * Y + Z * Z);//归一化
    Y = Y / sum;
    X = X / sum;
    Z = Z / sum;
 
    auto phi = atan2(Z, X);
    auto theta = asin(Y);
    u = 1- (phi + pi) / (2 * pi);
    v = (theta+pi/2 ) / pi;
    //cout << u << " " << v << endl;
    int j = int((u)*nx);//求出像素索引
    int i = int((1-v) * ny - 0.001f);
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > nx - 1) i = nx - 1;
    if (j > ny - 1) j = ny - 1;
 //返回对应贴图号的rgb的值
    if (k == 1) {
        r = ((float)Rr1[i][j]) / 255.0;
        g = ((float)Gg1[i][j]) / 255.0;
        b = ((float)Bb1[i][j]) / 255.0;

   }
    else if (k == 2) {
        r = ((float)Rr2[i][j]) / 255.0;
        g = ((float)Gg2[i][j]) / 255.0;
        b = ((float)Bb2[i][j]) / 255.0;
    }
   
   
    //cout <<i<<" "<<j<<" " << Rr[i][j] << "  " << Gg[i][j] << " " << r << "  " << g << "  " << b << endl;
    return vec3{r,g,b};

}
//射线和空间内的物体相交
bool scene_intersect(const vec3 &orig, const vec3 &dir, const std::vector<Box> &box, const std::vector<Sphere> &spheres, vec3 &hit, vec3 &N, Material &material)
{
    float spheres_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < spheres.size(); i++)
    {
        float dist_i;//当前深度 
        if(i==2)//这个贴图是木质贴图
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist)
        {
            spheres_dist = dist_i;
            hit = orig + dir * dist_i;
            N = (hit - spheres[i].center).normalize();
            material = spheres[i].material;
            material.diffuse_color = maps_picture(hit, spheres[i].center,1);
            continue;
        }
        if (i == 4)//这个贴图是地球贴图
            if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist)
            {
                spheres_dist = dist_i;
                hit = orig + dir * dist_i;
                N = (hit - spheres[i].center).normalize();
                material = spheres[i].material;
                material.diffuse_color = maps_picture(hit, spheres[i].center,2);
                continue;
            }
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist)
        {
            spheres_dist = dist_i;
            hit = orig + dir * dist_i;
            N = (hit - spheres[i].center).normalize();
            material = spheres[i].material;
        } 
    }
    //box投射
    float box_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < box.size(); i++)
    {
        float dist_i1;//当前深度 
        if (box[i].ray_intersect(orig, dir, dist_i1) && dist_i1 < box_dist&&dist_i1<spheres_dist)
        {
            box_dist = dist_i1;
            hit = orig + dir * dist_i1;
            if (abs(hit.x - box[i].bounds[0].x)<0.01)
                N = vec3{ -1,0,0 };
            else if (abs(hit.x - box[i].bounds[1].x) < 0.01)
                N = vec3{ 1,0,0 };
            else if (abs(hit.y - box[i].bounds[0].y) < 0.01)
                N = vec3{ 0,-1,0 };
            else if (abs(hit.y - box[i].bounds[1].y) < 0.01)
                N = vec3{ 0,1,0 };
            else if (abs(hit.z - box[i].bounds[0].z) < 0.01)
                N = vec3{ 0,0,-1 };
            else if(abs(hit.z - box[i].bounds[1].z) < 0.01)
                N = vec3{ 0,0,1 };
            material = box[i].material;

        }

    }
    //在z=-45处加棋盘格面
    float checkerboard_dist = std::numeric_limits<float>::max();
    if (std::abs(dir.z) > 1e-3)//避免垂直入射棋盘面
    {                                    // avoid division by zero
        float d = -(orig.z + 45) / dir.z; // 棋盘面是z=-45
        vec3 pt = orig + dir * d;//交点坐标
        if (d > 1e-3 && fabs(pt.x) < 100 && pt.y < 300 && pt.y > -300 && d < spheres_dist && d < box_dist)
        {
            checkerboard_dist = d;
            hit = pt;
            N = vec3{ 0, 0, 1 };
            material.diffuse_color = (int(.5 * hit.x + 1000) + int(.5 * hit.y)) & 1 ? vec3{ .3, .3, .3 } : vec3{ .3, .2, .1 };
        }
    }
   
    return std::min(std::min(box_dist, checkerboard_dist),spheres_dist) < 1000;//看是不是背景
}
//投射光线，返回颜色值，输入光源坐标，方向，球类数组，光源数组，迭代层数
vec3 cast_ray(const vec3 &orig, const vec3 &dir, const std::vector<Box>& box,const std::vector<Sphere> &spheres, const std::vector<Light> &lights, size_t depth = 0)
{
    vec3 point, N;//交点，法线方向
    Material material;
    if (depth > 4||!scene_intersect(orig, dir, box,spheres, point, N, material))
    {
        return vec3{0.48627,0.88627,0.69019}; // 返回背景色，浅绿色
    }
    //反射
    vec3 reflect_dir = reflect(dir, N).normalize();//反射方向
    vec3 reflect_orig = point + N*1e-3; // 沿着法线方向偏移一定距离，避免与自己相加
    vec3 reflect_color = cast_ray(reflect_orig, reflect_dir, box,spheres, lights, depth + 1);
    //折射
    vec3 refract_dir = refract(dir, N, material.refractive_index).normalize();
    vec3 refract_orig = refract_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    vec3 refract_color = cast_ray(refract_orig, refract_dir, box,spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;//漫反射强度，镜面反射强度
    for (size_t i = 0; i < lights.size(); i++)//遍历光源
    {
        vec3 light_dir = (lights[i].position - point).normalize();//光的传播方向，是这个光源和交点的连线
        float light_distance = (lights[i].position - point).norm();//光源和交点的距离

        vec3 shadow_orig = light_dir * N < 0 ? point : point + N * 1e-3; // 为了防止下一步中自己和自己相交，进行偏移处理
        vec3 shadow_pt, shadow_N;//
        Material tmpmaterial;
        //如果被遮挡则不进行该光源的光照计算，直接进行下一轮光源的计算，判断这个交点是否在这个光源的阴影中
        if (scene_intersect(shadow_orig, light_dir, box,spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir * N);//phong模型中的理想漫反射，与视点无关
        specular_light_intensity += powf(std::max(0.f, reflect(light_dir, N) * dir), material.specular_exponent) * lights[i].intensity;//镜面反射（高光）
    }
    //cout << N << endl;
    //cout << material.diffuse_color * diffuse_light_intensity * material.albedo[0] + vec3{ 1., 1., 1. } *specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2] + refract_color * material.albedo[3]<<endl;
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + vec3{1., 1., 1.} * specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2] + refract_color * material.albedo[3];
    // //颜色*强度*系数
}
//绘制

void render(const std::vector<Box>& box,const std::vector<Sphere> &spheres, const std::vector<Light> &lights)
{
    const int width = 960;                                                                                                   
    const int height = 600;
    const float fov = 1.05;//俯仰角
    std::vector<vec3> framebuffer(width * height);

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {//前两层循环，便利每一个像素点
            float x = (i + 0.5) - width / 2.;
            float y = -(j + 0.5) + height / 2.;
            float z = -height / (2. * tan(fov / 2.));
            vec3 dir = vec3{x, y, z}.normalize();
            framebuffer[i + j * width] = cast_ray(vec3{0, 0, 0}, dir, box,spheres, lights);
        }
    }

    std::ofstream ofs; 
    ofs.open("./out.ppm", std::ios::binary);
    ofs << "P6\n"
        << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height * width; ++i)
    {
        vec3 &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max > 1)
            c = c * (1. / max);
        for (size_t j = 0; j < 3; j++)
        {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main()
{
    new_img1 = imread("C:\\Users\\lh\\Desktop\\code\\wood.jpg");
    new_img2 = imread("C:\\Users\\lh\\Desktop\\code\\earthmap.jpg");
    read_rgb();
    //cout << "color" << Rr[691][108] << endl;
    //折射率，光的各部分权重, 光源，高光，反射，折射，材料原色，反射系数
    Material purpel_material(1.0, vec4{0.4, 0, 0, 0.0}, vec3{0.58, 0.44, 0.86}, 50);
    Material red_material(1.0, vec4{0.3, 0.1, 0.0, 0.0}, vec3{1, 0,0}, 10);
    Material mirror(1.0, vec4{0.0, 10.0, 0.8, 0.0}, vec3{1.0, 1.0, 1.0}, 1425);
    Material glass(1.5, vec4{0.0, 0.5, 0.1, 0.8}, vec3{0.6, 0.7, 0.8}, 125);

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(vec3{-3, 0, -16}, 2, purpel_material));
    spheres.push_back(Sphere(vec3{-1.0, -1.5, -12}, 2, glass));
    spheres.push_back(Sphere(vec3{6, -0.5, -18}, 3, red_material));
    spheres.push_back(Sphere(vec3{-7, 5, -18}, 4, mirror));
    spheres.push_back(Sphere(vec3{ -9, -5, -18 }, 3, red_material));

    std::vector<Box> box;
    box.push_back(Box(vec3{ 8, 3.5, -18 }, vec3{ 10, 5.5, -16 }, red_material));

    std::vector<Light> lights;
    lights.push_back(Light(vec3{-20, 20, 20}, 1.5));
    lights.push_back(Light(vec3{30, 50, -25}, 1.8));
    lights.push_back(Light(vec3{30, 20, 30}, 1.7));

    render(box,spheres, lights);
    return 0;
}
