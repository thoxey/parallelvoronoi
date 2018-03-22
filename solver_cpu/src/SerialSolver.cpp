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
std::vector<vec3> SerialSolver::makeDiagram(vec2 _imageDims, uint _numCells)
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
