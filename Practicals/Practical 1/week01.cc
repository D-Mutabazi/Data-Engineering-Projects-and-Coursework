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

    typedef int T;                  // The type of the values that the RVs can take on
    typedef DiscreteTable<T> DT;    // DT now is a short-hand for DiscreteTable<int>
    double defProb = 0.0;           // Any unspecified probs will default to this.
    rcptr< vector<T> > binDom (     // Lists the values that a particular RV can take on
        new vector<T>{0,1});
        
    rcptr< vector<T> > ZDom (     // Lists the values that a particular RV can take on
	new vector<T>{0,1,2});

    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines two RV ids: the id of X is 0
    // and the id of Y is 1. This is easy enough in very simple
    // problems, for more complex situations involving many RVs this
    // becomes cumbersome and we will need a datastructure such as a
    // map to save the RV ids in. Consult the userguide for more on
    // this.

    enum{X, Y, Z};

    //*********************************************************
    // Set up a discrete factor (in several ways) over two binary
    // RVs specifying that they must have odd parity (i.e. their
    // values will always differ).
    //***************************************************


    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // The most direct declaration. We show this as an example of
    // construction with a basic set of parameters. See the class
    // specific constructor from line 109 in
    // src/emdw-factors/discretetable.hpp for more detail on the
    // exact types of each variable.
    //
    // IMPORTANT: However, you will instead use a dynamic
    // declaration (lower down), because that will allow you to
    // access via its abstract category namely a Factor.
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 
    std::cout << " " << std::endl;
    std::cout << "**************QUESTION 5A***************" << std::endl;
    std::cout << " " << std::endl;
        
    std::cout << " i) " << std::endl;

    rcptr<Factor>ptrX =
     uniqptr<DT>(
      new DT(
       {X} ,
       {binDom} ,
       defProb,    // why is this present, the only outcome I need is that of whats known, unless this
       		   // is how the constructor Discret table is initialised OR STRUCTURED in EMDW?
       {
        {{0}, 0.5},
	{{1}, 0.5},
       } )) ; 
     
     std::cout <<__FILE__<<" : "<<__LINE__<<" : "<<*ptrX<<std::endl ;
  
   std::cout <<"ii) "<<std::endl;
   
   rcptr<Factor>ptrY=
    uniqptr<DT>(
     new DT(
      {Y},
      {binDom},
      defProb,
      {
       {{0}, 0.5},
       {{1}, 0.5},
      }));
    
    std::cout <<__FILE__<<" : "<<__LINE__<<" : "<<*ptrY<<std::endl;

   std::cout<<" iii) "<<endl ;
   
   rcptr<Factor>ptrZgXY = 
    uniqptr<DT>(
     new DT(
      {X,Y,Z},
      {binDom, binDom,ZDom}, //domain of X, Y, Z
      defProb,               // every other combination, factore/prob  = 0;
      {
       {{0, 0, 0}, 1},
       {{0, 1, 1}, 1},
       {{1, 0, 1}, 1},
       {{1, 1, 2}, 1},
      })) ;     
    std::cout<<__FILE__<<" : "<<__LINE__<<" : "<<*ptrZgXY<<std::endl ;
    
    
    std::cout << " " << std::endl;
    std::cout << "**************QUESTION 5C***************" << std::endl;
    std::cout << " " << std::endl;
    
    std::cout<<__FILE__<<" : "<<__LINE__<<" : "<<*ptrX->absorb(ptrY)->absorb(ptrZgXY)<<std::endl ;
    
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
