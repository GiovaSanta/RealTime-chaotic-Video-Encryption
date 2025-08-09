#include "../include/prbg_main_plcm.hpp"

std::vector<double> generatePRBGMainKeys(double x0, double p, int numParameters4subsequentPRBGas, uint64_t *sc) {

    //parameters4subsequentPRBGas , each future PRBGa will be fed with a key (x0) and p (controlParameter)
    std::vector<double> keysAndControlPs ;

    double xi = x0 ;

    //implementing the PRBG of main thread (pseudo Random Bit Generator) through the use of the PieceWiseLinearChaoticMap (PLCM) described by the research article
    //it is by default sequential, look into in the future of can be found a method to implement the PRBG through another method which can be parallelizable
    //as the PRBG based on the PLCM is nonParallelizable by default

    for ( int i = 0 ; i < numParameters4subsequentPRBGas ; i++ ) {
        if ( xi >= 0 && xi < p ) {
            xi = xi / p ;
        } 
        else if( xi >= p && xi <= 0.5 ){
            xi = (xi - p) / (0.5 - p) ;
        }
        else if( xi > 0.5 && xi <= 1.0 ){
            //originally in the paper there is a recursive call but as that recursion is always one stepped lets say... 
            //...so we can just avoid the recursion and instead write the checks again that the onestep recursion would have done
            xi = xi > 1.0 ? 1.0 : xi ; //we are clamping the value to 1 just in case
            xi = xi < 0.0 ? 0.0 : xi ; //similar if ever the value is negative
            xi = 1.0 - xi ;
            if ( xi >= 0 && xi < p){
                xi = xi / p;
            }
            else if ( xi >= p && xi <= 0.5){
                xi = (xi - p) / ( 0.5 -p );
            }
        }
        if( i < numParameters4subsequentPRBGas - 1 ){
            keysAndControlPs.push_back(xi);
            std::cout << "PRBG[ " << i << "] = " << keysAndControlPs[i] << std::endl; //debugging purposes
        } else {
            //*sc = xi ;
            *sc = static_cast<uint64_t>(xi * 1000000000) ;  
            //*sc = 1000000000 ;  //if i increase for one more order of magnitude then this one, encryption of frame breaks
            //*sc = 1000000;
            //*sc=10;
            std::cout << "PRBG[ " << i << "] = " << *sc << " (this is the global sc value)" << std::endl; //debugging purposes
        } 
    }
    return keysAndControlPs;
    
}