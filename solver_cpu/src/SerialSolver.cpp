/** File:    MacStableSolver.cpp
 ** Author:  Dongli Zhang
 ** Contact: dongli.zhang0129@gmail.com
 **
 ** Copyright (C) Dongli Zhang 2013
 **
 ** This program is free software;  you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation; either version 2 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY;  without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
 ** the GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program;  if not, write to the Free Software
 ** Foundation, 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "SerialSolver.h"

//----------------------------------------------------------------------------------------------------------------------

SerialSolver::SerialSolver()
{

}
SerialSolver::~SerialSolver()
{

}
std::vector<vec3> SerialSolver::makeDiagram_brute(vec2 _imageDims, uint _numCells)
{
    std::vector<vec3> ret;
    uint pixCount = _imageDims.x * _imageDims.y;
    ret.reserve(pixCount);

    std::vector<vec2> cellPos;
    cellPos.reserve(_numCells);
    std::vector<vec3> cellColour;
    cellColour.reserve(_numCells);

    for(uint i = 0; i < _numCells; i++)
    {
        cellPos[i] = vec2(utils::randRange(_imageDims.x), utils::randRange(_imageDims.y));
        cellColour[i] = vec3(utils::randRange(255), utils::randRange(255), utils::randRange(255));
    }

    uint w = _imageDims.x;
    uint h = _imageDims.y;
    uint d = 0;

    for (uint hh = 0; hh < h; hh++)
    {
        for (uint ww = 0; ww < w; ww++)
        {
            int ind = -1;
            uint dist = INT32_MAX;
            for (size_t it = 0; it < _numCells; it++)
            {
                d = utils::DistanceSqrd(cellPos[it], vec2(ww,hh));
                if (d < dist)
                {
                    dist = d;
                    ind = it;
                }
            }

            if (ind > -1)
                ret.push_back(cellColour[ind]);
            //ret[utils::get2DIndex(_imageDims.x, vec2(ww, hh))] = cellColour[ind];
        }
    }
    return ret;
}

bool SerialSolver::comparator(const std::pair<int,int> &A,const std::pair<int,int> &B)
{
    if(A.second<=B.second)
    {
        if(A.first>=B.first)
            return 1;
        else return 0;
    }
    return 0;
}

uint SerialSolver::getHashVal(vec2 _cellPos, const uint _GRID_RES, vec2 _imageDims, vec2 _offset)
{
    uint xfrac = (_cellPos.x/_imageDims.x) * _GRID_RES;
    uint yfrac = (_cellPos.y/_imageDims.x) * _GRID_RES;

    uint gridPos[2] = {xfrac, yfrac};

    return gridPos[0]+_offset.x * _GRID_RES + gridPos[1]+_offset.y;
}

std::vector<vec3> SerialSolver::makeDiagram_NN(vec2 _imageDims, uint _numCells)
{
    const uint GRID_RES = 8;

    std::vector<vec3> ret;
    uint pixCount = _imageDims.x * _imageDims.y;
    ret.reserve(pixCount);

    std::vector<vec2> cellPos;
    cellPos.reserve(_numCells);
    std::vector<vec3> cellColour;
    cellColour.reserve(_numCells);

    std::vector<uint> hash(_numCells, 0);

    for(uint i = 0; i < _numCells; i++)
    {
        cellPos[i] = vec2(utils::randRange(_imageDims.x), utils::randRange(_imageDims.y));
        cellColour[i] = vec3(utils::randRange(255), utils::randRange(255), utils::randRange(255));
    }

    for(uint i = 0; i<_numCells; i++)
    {
        hash[i] = getHashVal(cellPos[i], GRID_RES, _imageDims);
    }

    std::multimap<uint, vec2> hashToPosMap;
    for (uint i = 0; i < _numCells; ++i)
        hashToPosMap.insert(std::make_pair(hash[i], cellPos[i]));
    std::multimap<uint, uint> posToColourKeyMap;
    for (uint i = 0; i < _numCells; ++i)
        posToColourKeyMap.insert(std::make_pair(cellPos[i].x, i));

    uint w = _imageDims.x;
    uint h = _imageDims.y;
    uint d = 0;


    for (uint hh = 0; hh < h; hh++)
    {
        for (uint ww = 0; ww < w; ww++)
        {
            int ind = -1;
            uint dist = INT32_MAX;

            for(int i = -1; i < 2; i++)
                for(int j = -1; j < 2; j++)
                {
                    uint hashVal = getHashVal(vec2(ww,hh), GRID_RES, _imageDims, vec2(i,j));
                    if(hashVal >= GRID_RES*GRID_RES)//uint cant be < 0
                        continue;
                    auto itlow = hashToPosMap.lower_bound(hashVal);
                    auto itup = hashToPosMap.upper_bound(hashVal);
                    for (auto it=itlow; it!=itup; ++it)
                    {
                        d = utils::DistanceSqrd((*it).second, vec2(ww,hh));
                        if (d < dist)
                        {
                            dist = d;
                            ind = posToColourKeyMap.find((*it).second.x)->second;
                        }
                    }
                }
            if (ind > -1)
                ret.push_back(cellColour[ind]);
        }
    }

    return ret;
}
