#pragma once

#ifndef PBRTSTRUCT_H
#define PBRTSTRUCT_H

#include <string>
#include <map>
#include <vector>

struct PBRTParameter {
    std::string type;
    std::vector<std::string> values;
};


struct PBRTStatement {
    std::string identifier;
    std::string type;
    std::map<std::string, PBRTParameter> parameter_list;
};

#endif