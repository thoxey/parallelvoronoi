#ifndef UTILS_H
#define UTILS_H


#include<algorithm>
#include<vector>
#include<unordered_map>
#include<cmath>
#include<random>
#include<iostream>
#include <algorithm>

#include<glm/vec3.hpp>
#include<glm/vec2.hpp>

//----------------------------------------------------------------------------------------------------------------------
/// @file utils.h
/// @brief The utility header, contains definitions for grid components and universal functions
/// @author Tom Hoxey
/// @version 1.0
/// @date
//----------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------
/// TYPEDEFS TO ALLOW FOR EASILY SWAPPING BETWEEN VECTOR IMPLEMENTATIONS
#define USE_DOUBLE_PRECISION
//-fsingle-precision-constant
#ifdef USE_DOUBLE_PRECISION
typedef double real;
typedef glm::dvec3 vec3;
typedef glm::dvec2 vec2;
#else
typedef float real;
typedef glm::vec3 vec3;
typedef glm::vec2 vec2;
#endif
typedef glm::uvec3 uvec3;
typedef glm::uvec2 uvec2;
typedef glm::ivec3 ivec3;
typedef unsigned int uint;
//----------------------------------------------------------------------------------------------------------------------
const vec3 rightVec = vec3(1.0,0.0,0.0);
const vec3 leftVec = vec3(-1.0,0.0,0.0);
const vec3 upVec = vec3(0.0,1.0,0.0);
const vec3 downVec = vec3(0.0,-1.0,0.0);
const vec3 forwardVec = vec3(0.0,0.0,1.0);
const vec3 backwardVec = vec3(0.0,0.0,1.0);

const uvec3 urightVec = uvec3(1,0,0);
const uvec3 uleftVec = uvec3(-1,0,0);
const uvec3 uupVec = uvec3(0,1,0);
const uvec3 udownVec = uvec3(0,-1,0);
const uvec3 uforwardVec = uvec3(0,0,1);
const uvec3 ubackwardVec = uvec3(0,0,-1);
//----------------------------------------------------------------------------------------------------------------------
namespace utils
{
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the index of a cell depending on the size of the grid
/// @param vec2 _length :
/// @param vec2 _pos :
//----------------------------------------------------------------------------------------------------------------------
uint DistanceSqrd(vec2 p1, vec2 p2);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the index of a cell depending on the size of the grid
/// @param uint _length : The amount of cells in a given axis of the grid
/// @param uvec3 _pos : The position of he cell in the grid
//----------------------------------------------------------------------------------------------------------------------
uint get2DIndex(uint _length, uvec2 _pos);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the index of a cell depending on the size of the grid
/// @param uint _length : The amount of cells in a given axis of the grid
/// @param uvec3 _pos : The position of he cell in the grid
//----------------------------------------------------------------------------------------------------------------------
uint get3DIndex(uint _length, uvec3 _pos);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns a random number between 0 and _max
/// @param real _max : The maximum size of the returned value
//----------------------------------------------------------------------------------------------------------------------
real randRange(real _max);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns a random number between _min and _max
/// @param real _min : The minimum size of the returned value
/// @param real _max : The maximum size of the returned value
//----------------------------------------------------------------------------------------------------------------------
real randRange(real _min, real _max);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns a value at _x between _ and _b
/// @param real _a : The point to interpolate from
/// @param real _b : The point to interpolate to
/// @param real _x : The amount to interpolate
//----------------------------------------------------------------------------------------------------------------------
real lerp(real _a, real _b, real _x);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns the amount interpolated between a and b if x is l away
/// @param real _a : The point to interpolate from
/// @param real _b : The point to interpolate to
/// @param real _l : The length interpolated between a and b
//----------------------------------------------------------------------------------------------------------------------
real invLerp(real _a, real _b, real _l);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Returns a value at position xyz based on the values at the verticies
/// @param real * V : The values at each the verticies
/// @param real x : The x position of the point we are querying
/// @param real y : The y position of the point we are querying
/// @param real z : The z position of the point we are querying
//----------------------------------------------------------------------------------------------------------------------
real trilerp(real * V, real x, real y, real z);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Checks if a point a is in the bounds of b and c
/// @param vec3 _a : The point we are querying
/// @param uvec3 _b : The bottom corner of the fluid cells (0,0,0)
/// @param uvec3 _c : The top corner of the fluid cells (1,1,1)
//----------------------------------------------------------------------------------------------------------------------
bool isInBounds(vec3 _a, vec3 _b, vec3 _c);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Prints the vector inputted
/// @param uvec3 _x : The vector to print
/// @param string _message : An optional message to print after the vector
//----------------------------------------------------------------------------------------------------------------------
void printvec(uvec3 _x, std::string _message  = "");
//----------------------------------------------------------------------------------------------------------------------
/// @brief Prints the vector inputted
/// @param vec3 _x : The vector to print
/// @param string _message : An optional message to print after the vector
//----------------------------------------------------------------------------------------------------------------------
void printvec(vec3 _x, std::string _message  = "");
}




//Empty doc tags for copy pasting!
//----------------------------------------------------------------------------------------------------------------------
/// @brief
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
/// @brief
/// @param
//----------------------------------------------------------------------------------------------------------------------


#endif // UTILS_H
