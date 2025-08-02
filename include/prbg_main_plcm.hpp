#ifndef PRBG_MAIN_PLCM_HPP
#define PRBG_MAIN_PLCM_HPP

#include <vector>
#include <iostream>

std::vector<double> generatePRBGMainKeys(double x0, double p, int numParameters4subsequentPRBGas, double *sc);

#endif