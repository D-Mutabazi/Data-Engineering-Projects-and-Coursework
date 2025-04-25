/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp"  // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "emdw.hpp"
#include "discretetable.hpp"
#include "clustergraph.hpp"
#include "lbp_cg.hpp"
#include "lbu_cg.hpp"
#include "messagequeue.hpp"

// standard headers
#include <iostream>  // cout, endl, flush, cin, cerr
#include <cctype>  // toupperZ
#include <string>  // string
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

// MVG factor dependencies
#include "prlite_genmat.hpp"
#include "sqrtmvg.hpp"

// These are to avoid doing std:: and emdw:: all the time. Don't do
// this in an .hpp file
using namespace std;
using namespace emdw;

//##################################################################
// Some example code. To compile this, go to the emdw/build
// directory and do a:
// cmake ../; make -j7 example
// To run this while in the build directory, do a:
// src/pmr/example
//
// For your own stuff, make a copy of this one to start with. Then
// edit the CMakeLists.txt (also in this directory) by adding your
// new target in the same way as this example.
//##################################################################

using namespace std;
using namespace emdw;

int main(int, char *argv[]) {

  // NOTE: this activates logging and unit tests
  initLogging(argv[0]);
  prlite::TestCase::runAllTests();

  try {

    //*********************************************************
    // Some random generator seeding. Just keep this as is
    //*********************************************************

    unsigned seedVal = emdw::randomEngine.getSeedVal();
    cout <<  seedVal << endl;
    emdw::randomEngine.setSeedVal(seedVal);

    /*****************************************************************
     * Predefine some types and constants
     *****************************************************************/

      typedef double T;                  // The type of the values that the RVs can take on
      typedef DiscreteTable<T> DT;    // DT now is a short-hand for DiscreteTable<int>
      rcptr< vector<T> > binDom (     // Lists the values that a particular RV can take on
              new vector<T>{0,1});

      //*********************************************************
      // Define the RVs
      //*********************************************************

      enum{x1, x2, x3,x4};

      // Gaussian factor dependencies
      typedef SqrtMVG SG;


      /*****************************************************************
       * set up the factors
       *****************************************************************/
      // factor (x1, x2)
      prlite::ColVector<double> mn_x1_x2(2);
      mn_x1_x2[0] = 1.0 ; 
      mn_x1_x2[1] = 2.0  ;
      prlite::RowMatrix<double> cov_x1_x2(2,2);
      cov_x1_x2(0,0) = 1.0;
      cov_x1_x2(0,1) = 2.0 ;
      cov_x1_x2(1,0) = 2.0 ;
      cov_x1_x2(1,1) = 5.0 ;
      rcptr<Factor> ptr_X1_X2(uniqptr<SG>(new SG({x1,x2}, mn_x1_x2, cov_x1_x2)));
      
      // factor (x2, x3)
      prlite::ColVector<double> mn_x2_x3(2);
      mn_x2_x3[0] = 3.0 ; 
      mn_x2_x3[1] = 4.0  ;
      prlite::RowMatrix<double> cov_x2_x3(2,2);
      cov_x2_x3(0,0) = 2.0;
      cov_x2_x3(0,1) = 2.0 ;
      cov_x2_x3(1,0) = 2.0 ;
      cov_x2_x3(1,1) = 3.0 ;
      rcptr<Factor> ptr_X2_X3(uniqptr<SG>(new SG({x2,x3}, mn_x2_x3, cov_x2_x3)));
    
      // stick gaussian factors into the array
      std::vector< rcptr<Factor> > factors;
      factors.push_back(ptr_X1_X2 );
      factors.push_back(ptr_X2_X3 );

      /*****************************************************************
       * observed data goes into this map
       *****************************************************************/

      std::map<emdw::RVIdType, AnyType> obsv;
      obsv.clear();
      // NOTE: The T() type conversion is important - otherwise the
      // AnyType will store an int which is wrong
      obsv[x1] = T(0.0);
     

      /*****************************************************************
       * put together the LTRIP cluster graph
       *****************************************************************/

      // for factor graph, use BETHE instead of LTRIP
      // for junction tree, use JTREE instead of LTRIP
      ClusterGraph cg(ClusterGraph::BETHE, factors, obsv);
      //cout << cg << endl;

      // export the graph to graphviz .dot format
      cg.exportToGraphViz("week7_Gaussian_factors");
      

      /*****************************************************************
       * calibrate the graph using LBP
       *****************************************************************/

      map<Idx2, rcptr<Factor> > msgs;
      MessageQueue msgQ;

      // choose either BP or BU
      msgs.clear();
      msgQ.clear();
      // loopyBP_CG implements the loopy belief propagation
      // (Shafer-Shenoy) algorithm. Using loopyBU_CG instead results
      // in the loopy belief update (Lauritzen-Spiegelhalter)
      // algorithm.
      unsigned nMsgs = loopyBP_CG(cg, msgs, msgQ);
      cout << "Sent " << nMsgs << " messages before convergence\n";

      /*****************************************************************
       * query the graph using LBP
       *****************************************************************/

      // lets check out the individually decoded bits
      cout << "After decoding the probs that p(x2, x3|x1=0)  \n";

      // queryLBU_CG here.
      rcptr<Factor> qPtr = queryLBP_CG(cg, msgs, {x2, x3})->normalize();
      // downcast the factor to use additional methods
      rcptr<SG> dwnqPtr; dwnqPtr =  dynamic_pointer_cast<SG>(qPtr) ; 
      std::cout << "Mean vector (x2, x3| x1=0): " << dwnqPtr->getMean() << "\nCovariance matrix (x2, x3| x1=0): " << dwnqPtr->getCov()<< std::endl;
      std::cout << "Information matrix (x2, x3| x1=0): " << dwnqPtr->getK() << "\nInformation vector (x2 , x3| x1=0): " << dwnqPtr->getH() << std::endl;
    


      // for (RVIdType var = b0; var <= b6; var++) {
      //   if (obsv.find(var) == obsv.end()) {
      //     // If you calibrated with loopyBU_CG, you will have to use
      //     // queryLBU_CG here.
      //     rcptr<Factor> qPtr = queryLBP_CG(cg, msgs, {var})->normalize();
      //     //cout << *qPtr << endl;
      //     double p1 = qPtr->potentialAt({var}, {T(1)});
      //     cout << p1 << " ";
      //     //cout << (p1 < 0.5 ? 0 : 1) << " ";
      //   } // if
      //   else { cout << T(obsv[var]) << " "; }
      // } // for
      // cout << endl;

      return 0; // tell the world that all is fine
  } // try

  catch (char msg[]) {
    cerr << msg << endl;
  } // catch

  // catch (char const* msg) {
  //   cerr << msg << endl;
  // } // catch

  catch (const string& msg) {
    cerr << msg << endl;
    throw;
  } // catch

  catch (const exception& e) {
    cerr << "Unhandled exception: " << e.what() << endl;
    throw e;
  } // catch

  catch(...) {
    cerr << "An unknown exception / error occurred\n";
    throw;
  } // catch

} // main
