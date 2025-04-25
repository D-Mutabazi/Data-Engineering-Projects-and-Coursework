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

// standard headers
#include <iostream>  // cout, endl, flush, cin, cerr
#include <cctype>  // toupper
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

    //*********************************************************
    // Predefine some types and constants
    //*********************************************************    

    // Gaussian factor dependencies
    typedef SqrtMVG SG;

    enum{x1, x2, x3,x4, x, y, z, s};

    /* 3: Practical: Modelling and inference for a Gaussian Markov network */
    // a) {on paper}

    // b)
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
    
    // access abstract class of x1x2 gaussian factor
    rcptr<SG> dwnPtr; dwnPtr = dynamic_pointer_cast<SG>(ptr_X1_X2);
    std::cout << "Mean vector (x1, x2): " << dwnPtr->getMean() << std::endl;
    std::cout << "Covariance matrix (x1, x2): " << dwnPtr->getCov() << std::endl;
    std::cout << "Information matrix (x1, x2): " << dwnPtr->getK() << std::endl;
    std::cout << "Information vector (x1,x2): " << dwnPtr->getH() << std::endl;

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

    // access abstract class of x2x3 gaussian factor
    rcptr<SG> dwnPtr_x2_x3; dwnPtr_x2_x3 = dynamic_pointer_cast<SG>(ptr_X2_X3);
    std::cout << "Mean vector (x2,x3): " << dwnPtr_x2_x3->getMean() << std::endl;
    std::cout << "Covariance matrix (x2, x3): " << dwnPtr_x2_x3->getCov() << std::endl;
    std::cout << "Information matrix (x2, x3): " << dwnPtr_x2_x3->getK() << std::endl;
    std::cout << "Information vector (x2, x3): " << dwnPtr_x2_x3->getH() << std::endl;

    // joint distribution
    rcptr<Factor> ptr_X1_X2_X3 = ptr_X1_X2->absorb(ptr_X2_X3);
    rcptr<SG> dwnPtr_x1_x2_x3; dwnPtr_x1_x2_x3 =  dynamic_pointer_cast<SG>(ptr_X1_X2_X3) ;  
    std::cout << "Mean vector (x1, x2, x3): " << dwnPtr_x1_x2_x3->getMean() << std::endl;
    std::cout << "Covariance matrix (x1, x2, x3): " << dwnPtr_x1_x2_x3->getCov() << std::endl;
    std::cout << "Information matrix (x1, x2, x3): " << dwnPtr_x1_x2_x3->getK() << std::endl;
    std::cout << "Information vector (x1, x2, x3): " << dwnPtr_x1_x2_x3->getH() << std::endl;
    //i) X1 & X3 are statistically independent

    // c) {on paper}

    // d) p(x1, x3)
    rcptr<Factor> ptr_X1_X3 = ptr_X1_X2_X3->marginalize({x1, x3}) ;
    rcptr<SG> downPtr_x1_x3; downPtr_x1_x3 = dynamic_pointer_cast<SG>(ptr_X1_X3) ;
    std::cout << "Mean vector (x1, x3): " << downPtr_x1_x3->getMean() << "\nCovariance matrix (x1, x3): " << downPtr_x1_x3->getCov()<< std::endl;
    std::cout << "Information matrix (x1, x3): " << downPtr_x1_x3->getK() << "\nInformation vector (x1 x3): " << downPtr_x1_x3->getH() << std::endl;

    //i) export + plot 2D mesh
    downPtr_x1_x3->export2DMesh("x1_x3_post_mesh.txt", x1, x3, 128); // ? 3rd argumment correct value?

    //e) {on paper}

    //f)
    rcptr<Factor> ptr_X1_X3_g_X2 = ptr_X1_X2_X3->observeAndReduce({x2}, {1.0}) ;
    rcptr<SG> downPtr_X1_X3_g_X2; downPtr_X1_X3_g_X2 = dynamic_pointer_cast<SG>(ptr_X1_X3_g_X2) ;
    std::cout << "Mean vector (x1, x3| x2=1): " << downPtr_X1_X3_g_X2->getMean() << "\nCovariance matrix (x1, x3 | x2=1): " << downPtr_X1_X3_g_X2->getCov()<< std::endl;
    std::cout << "Information matrix (x1, x3 | x2=1): " << downPtr_X1_X3_g_X2->getK() << "\nInformation vector (x1 , x3| x2=1): " << downPtr_X1_X3_g_X2->getH() << std::endl;

    //g)
    std::cout << "3G)"<< std::endl;

    /* ------------- Belief propagation ----------------*/
    // initialize cluser beliefs
    rcptr<Factor> cluster1_Belief_x1_x2 = ptr_X1_X2 ; 
    rcptr<Factor> cluster2_Belief_x2_x3 = ptr_X2_X3 ;

    // initialise messages: remember: variables are continous - vacous initialisation
    rcptr<Factor> msg1_2(uniqptr<SG>(new SG({x2}, 0, 0))) ; 
    rcptr<Factor> msg2_1(uniqptr<SG>(new SG({x2}, 0, 0))) ; 

    // initialie sepset beliefs + message updates
    rcptr <Factor> sepsetX2_belief = msg1_2->absorb(msg2_1) ;
    rcptr<Factor> msg1_2_update ;
    rcptr<Factor> msg2_1_update ;

    // message passing schedule: 1->2 && 2->1 (convergence should be immediate)
    /* cluster 1->2*/
    msg1_2_update = cluster1_Belief_x1_x2->marginalize({x2}) ; 
    
    /* cluser 2->1*/
    msg2_1_update = cluster2_Belief_x2_x3->marginalize({x2}) ;

    // updated sepset belief -  clique tree , thus should convegre immediately
    sepsetX2_belief = msg1_2_update->absorb(msg2_1_update) ;

    // updated cluster beliefs
    cluster1_Belief_x1_x2 =  cluster1_Belief_x1_x2->absorb(msg2_1_update) ;
    cluster2_Belief_x2_x3 = cluster2_Belief_x2_x3->absorb(msg1_2_update) ; 

    // Inference step
    /* posterior belief p(x2, x3|x1=0)*/
    rcptr<Factor> pX2_X2gX1 = cluster2_Belief_x2_x3->observeAndReduce({x1}, {0.0});
    rcptr<SG> dwnPtr_pX2_X2gX1; dwnPtr_pX2_X2gX1 =  dynamic_pointer_cast<SG>(pX2_X2gX1) ;  

    // std::cout << "3g) p(x2, x3|x1=0)" << *pX2_X2gX1 << std::endl ;

    std::cout << "Mean vector (x2, x3| x1=0): " << dwnPtr_pX2_X2gX1->getMean() << "\nCovariance matrix (x2, x3| x1=0): " << dwnPtr_pX2_X2gX1->getCov()<< std::endl;
    std::cout << "Information matrix (x2, x3| x1=0): " << dwnPtr_pX2_X2gX1->getK() << "\nInformation vector (x2 , x3| x1=0): " << dwnPtr_pX2_X2gX1->getH() << std::endl;

    std::cout << "3F) - CHECK LBP file"<< std::endl;

    /**** Question 4: Practical: Modelling and inference for a Gaussian Bayes Network  ***/
    std::cout << "4a) {In workbook}"<< std::endl;

    std::cout << "4b)"<< std::endl;
    // p(x) factor 
    // Step 1: Define p(x) as a univariate Gaussian with mean 0 and variance 4
    prlite::RowMatrix<double> var_x(1, 1);
    var_x(0, 0) = 4.0;
    prlite::ColVector<double> mn_x(1);  
    mn_x[0] = 0.0;
    rcptr<Factor> ptr_px(uniqptr<SG>(new SG({x}, mn_x, var_x))); // check emd/src/emdw-factors
    // Step 2: Set up affine transform Y = X + V
    // So A = 1, c = 0, noiseR = 0 (identity-like for simplicity)
    prlite::ColMatrix<double> A(1, 1); A(0, 0) = 1.0;
    prlite::ColVector<double> c(1);   c[0] = 0.0;
    prlite::ColMatrix<double> R(1, 1); R(0, 0) = 1.0;  // Almost noiseless identity, to avoid singularity

    // Step 3: Construct the joint Gaussian over (x, y) -> p(x, y)
    rcptr<Factor> ptr_jointXY(
      uniqptr<SG>(
        SG::constructAffineGaussian(*dynamic_pointer_cast<SG>(ptr_px), A, c, {y}, R)
      )
    );

    // observe y=1 + normalize, determine p(x| y=1)
    rcptr<Factor> ptrXgY= ptr_jointXY->observeAndReduce({y}, {1.0})->normalize() ;
    //downcast to point to SqrtMVG factor to extract mean and covariance
    rcptr<SG> dwnPtr_XgY; dwnPtr_XgY = dynamic_pointer_cast<SG>(ptrXgY);
    std::cout << "Mean vector p(x | y = 1 ): " << dwnPtr_XgY->getMean() << "\nCovariance matrix p(x | y = 1 ): " << dwnPtr_XgY->getCov()<< std::endl;
    std::cout << "Results are identical to hand calculations"<< std::endl;

    std::cout << "4c) {In workbook}"<< std::endl;

    std::cout << "4d) determine p(x,z | s = 5)"<< std::endl;
    // 1: p(z) SqrtMVG factor determine, p(z) ~ N(4, 10)
    prlite::RowMatrix<double> var_z(1, 1);
    var_z(0, 0) = 10.0;
    prlite::ColVector<double> mn_z(1);  
    mn_z[0] = 4.0;
    rcptr<Factor> ptr_pz(uniqptr<SG>(new SG({z}, mn_z, var_z))); // check emd/src/emdw-factors

    //2. determine p(x, z, s) : s = x + z + w, w ~ N(0, 0.5)
    // So A = [1, 1], c = 0, noiseR = 0.5 (identity-like for simplicity)
    prlite::ColMatrix<double> As(1, 2); // 1 row, 2 cols
    As(0, 0) = 1.0; // Coefficient for x
    As(0, 1) = 1.0; // Coefficient for z
    std::cout << "As matrix:\n"<< As<< std::endl;

    prlite::ColVector<double> cs(1);   cs[0] = 0.0;
    prlite::ColMatrix<double> Rs(1, 1); Rs(0, 0) = std::sqrt(0.5);  // Almost noiseless identity, to avoid singularity

    //3. joint gaussian over (x, y and z) => p(x, y , z) 
    rcptr<Factor> ptr_jointXZ= ptr_px->absorb(ptr_pz) ; // multiplying absorbs all the information
    rcptr<Factor> ptr_jointXZS(
      uniqptr<SG>(
        SG::constructAffineGaussian(*dynamic_pointer_cast<SG>(ptr_jointXZ), As, cs, {s}, Rs)
      )
    );
    //4. determine p(x,z |s  = 5)
    rcptr<Factor> ptr_XZgS = ptr_jointXZS->observeAndReduce({s}, {5.0})->normalize() ; 

    //5. canonical representation : downcast to pointer to sqrtMVG class
    rcptr<SG> dwnPtr_XZgS; dwnPtr_XZgS = dynamic_pointer_cast<SG>(ptr_XZgS);
    std::cout << "Information matrix p(x, z| s=5): " << dwnPtr_XZgS->getK() << "\nInformation vector p(x , z| s=5): " << dwnPtr_XZgS->getH() << std::endl;
    std::cout << "**Results are identical to hand calculation**\n" << std::endl ;

    std::cout << "4e) {In workbook}"<< std::endl;

    std::cout << "4f) Determine p(x, z | y = 1, s = 5)"<< std::endl;

    //1. product of factors: p(x) xp(z) x p(y=1 |x) x p(s =5 | x, z) {absorb information from each factor}
    rcptr<Factor> ptr_XZgYS = ptrXgY->absorb(ptr_XZgS)->cancel(ptr_px)->normalize() ; // not p(x) multiplied twice, hence the division
    
    //2. marginalise out z: i.e. p(x | y = 1, s =5)
    rcptr<Factor> ptr_XgYS = ptr_XZgYS->marginalize({x})->normalize() ; 

    //3. Extract mean and covariance (by downcasting) and compare to hand calculations (4e)
    rcptr<SG> downPtr_XgYS; downPtr_XgYS = dynamic_pointer_cast<SG>(ptr_XgYS);
    std::cout << "Mean vector p(x| y=1, s=5): " << downPtr_XgYS->getMean() << "\nCovariance matrix p(x | y=1, s=5): " << downPtr_XgYS->getCov()<< std::endl;
    std::cout << "**Results are identical to hand calculation**\n" << std::endl ;




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
