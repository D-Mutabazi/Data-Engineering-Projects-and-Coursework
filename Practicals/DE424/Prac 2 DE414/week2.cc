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
    rcptr< vector<T> > varDom (     // Lists the values that a particular RV can take on
        new vector<T>{0,1,2});
        
    //*********************************************************
    // Define the RVs
    //*********************************************************

    // The enum statement here predefines 4 RV ids: the id of I is 0
    // and the id of C is 1, iD of M is 2 and the ID of R is 3

    enum{I, C, M, R};

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
    std::cout << "**************QUESTION 4***************" << std::endl;
    std::cout << " " << std::endl;
        
    std::cout << " i) " << std::endl;

    rcptr<Factor>ptrI =
     uniqptr<DT>(
      new DT(
       {I} ,
       {varDom} ,
       defProb,    // why is this present, the only outcome I need is that of whats known, unless this
       		   // is how the constructor Discret table is initialised OR STRUCTURED in EMDW?
       {
        {{0}, 1.0/3.0},
	      {{1}, 1.0/3.0},
        {{2,}, 1.0/3.0}
       } )) ; 
     
    std::cout <<__FILE__<<" : "<<__LINE__<<" : "<<*ptrI<<std::endl ;
  
    std::cout <<"ii) "<<std::endl;

    rcptr<Factor>ptrC = 
      uniqptr<DT>(
        new DT(
          {C},
          {varDom},
          defProb ,
          {
            {{0}, 1.0/3.0},
            {{1}, 1.0/3.0},
            {{2}, 1.0/3.0}
          }));

    std::cout <<__FILE__<<" : "<<__LINE__<<" : "<<*ptrC<<std::endl ;
  
    rcptr<Factor>ptrMgIC =
      uniqptr<DT>(
        new DT(
          {I, C , M},
          {varDom, varDom, varDom},
          defProb,
          {
            {{0, 0, 1}, 0.5},
            {{0, 0, 2}, 0.5},
            {{0, 1, 2}, 1} ,
            {{0, 2, 1}, 1},
            {{1, 0, 2}, 1} ,
            {{1, 1, 0}, 0.5},
            {{1, 1, 2}, 0.5},
            {{1, 2, 0}, 1},
            {{2, 0, 1}, 1} ,
            {{2, 1, 0}, 1},
            {{2, 2, 0}, 0.5},
            {{2, 2, 1}, 0.5} ,
          } 
        )
      );

   
    std::cout <<__FILE__<<" : "<<__LINE__<<" : "<<*ptrMgIC<<std::endl ;

    std::cout <<"QUESTION 4A - JOINT DISTRIBUTION P(I,M,C)"<<std::endl ;
    //joint distribution found but distribution to be normalized
    rcptr<Factor>ptrMIC = ptrI->absorb(ptrC)->absorb(ptrMgIC) ;

    std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrMIC<<std::endl ;
    
    
    std::cout <<"QUESTION 4.2(B) - condtional distribution P(M|I=0,M=0)"<<std::endl ;
    
    rcptr<Factor>ptrCgIE0ME1 = ptrMIC->observeAndReduce({I,M} ,{0, 1})->normalize() ; //conditioned on I = 0 and M =1 
                                                                                      // distribution normalized

    std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrCgIE0ME1<<std::endl ;


    std::cout <<"QUESTION 4.3(A) - Flow of influence via collider nodes"<<std::endl ;

    std::cout <<"i)"<<std::endl ;
    rcptr<Factor> ptrCgIE1MU = ptrMIC->observeAndReduce({I}, {1})  ;// observe value of I =1 
    ptrCgIE1MU = ptrCgIE1MU->marginalize({C})->normalize() ;  //marginalize over C if M unobserved and normalize distribution

    std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrCgIE1MU<<std::endl;

    std::cout <<"ii) -M to one of the values it can take on, while leaving the other variables unobserved"<<std::endl ;
    rcptr<Factor>ptrCgME0IU = ptrMIC->observeAndReduce({M}, {0}) ; // M given value = 0 , I,C unobserved
    ptrCgME0IU = ptrCgME0IU->marginalize({C})->normalize() ; //marginalize over C and normalize
    std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrCgME0IU<<std::endl;

    std::cout <<"ii) with M still on the value you chose, set I to one of the two remaining values it can legally take on"<<std::endl ;
    rcptr<Factor>ptrCgME0IE2 = ptrMIC->observeAndReduce({I,M}, {2,0}) ; // observe I =2, M= 0;
    ptrCgME0IE2 = ptrCgME0IE2->marginalize({C})->normalize(); // marginalize over C and normalize distribution

    std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrCgME0IE2<<std::endl;

    std::cout <<"QUESTION 4.3(B) - Flow of influence via collider nodes"<<std::endl ;

    //factor for P(R|M)
    rcptr<Factor>ptrRgM =
        uniqptr<DT>(
          new DT(
            {M, R},
            {varDom, varDom},
            defProb,
            {
              {{0, 0}, 0.8},
              {{0, 1}, 0.1},
              {{0, 2}, 0.1} ,
              {{1, 0}, 0.1},
              {{1, 1}, 0.8} ,
              {{1, 2}, 0.1},
              {{2, 0}, 0.1},
              {{2, 1}, 0.1},
              {{2, 2}, 0.8} ,
              
            } 
          )
        );
      ptrRgM  = ptrRgM->normalize() ; // noralize conditional distribution
      std::cout <<"bi)Set I to one of its allowed values, while leaving the other variables unobserved,  R unobserved"<<std::endl ;
      rcptr<Factor>ptrICMR = ptrI->absorb(ptrC)->absorb(ptrMgIC)->absorb(ptrRgM) ;       // joint distribution P(I,C,M,R)

      rcptr<Factor>ptrIE0RUMUCU  = ptrICMR->observeAndReduce({I}, {0}) ; // observe I = 0 , R= 1, C=M=unknown
      ptrIE0RUMUCU  = ptrIE0RUMUCU->marginalize({C})->normalize() ; // marginalize over C and normalize distribution
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrIE0RUMUCU<<std::endl;

      std::cout <<"bii)Set I to one of its allowed values, while leaving the other variables unobserved,  R observed"<<std::endl ;
      rcptr<Factor>ptrIE0RE1MUCU  = ptrICMR->observeAndReduce({I, R}, {0, 1}) ; // observe I = 0 , R= 1, C=M=unknown
      ptrIE0RE1MUCU  = ptrIE0RE1MUCU->marginalize({C})->normalize() ; // marginalize over C and marginalize distribution
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrIE0RE1MUCU<<std::endl;

      std::cout <<"QUESTION 4.3(C) - Flow of influence via collider nodes"<<std::endl ;



      std::cout <<"QUESTION 4.3(D) - Flow of influence via collider nodes"<<std::endl ;
      rcptr<Factor>ptrIERCUMU = ptrICMR->observeAndReduce({I, R}, {1, 1}) ; // observe I=R=1 and C=M=UNKNOWN
      ptrIERCUMU = ptrIERCUMU->marginalize({C})->normalize() ;  // marginalize over the value of C and normalize
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrIERCUMU<<std::endl;

      std::cout <<"QUESTION 4.4 Flow of influence via non-collider nodes"<<std::endl ;

      std::cout <<"A)"<<std::endl ;
      rcptr<Factor>ptrRgIE1MUCU = ptrICMR->observeAndReduce({I}, {0}) ; // observe I = 0 , rest of variables unobserved
      ptrRgIE1MUCU =  ptrRgIE1MUCU->marginalize({R})->normalize() ; // marginalize over R and normalize
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrRgIE1MUCU<<std::endl;

      std::cout <<"Bi)"<<std::endl ;
      rcptr<Factor>ptrME0REIECEU = ptrICMR->observeAndReduce({M}, {0}) ;  // observe M=0 , rest of variables unknown
      ptrME0REIECEU = ptrME0REIECEU->marginalize({R})->normalize(); // marginalize over R and normalize
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrME0REIECEU<<std::endl;

      std::cout <<"Bii)"<<std::endl ;
      rcptr<Factor>ptrME0REUIE2CEU = ptrICMR->observeAndReduce({I, M}, {2,0}) ;  // observe M=0, I=2 , rest of variables unknown
      ptrME0REUIE2CEU = ptrME0REUIE2CEU->marginalize({R})->normalize(); // marginalize over R and normalize
      std::cout<<__FILE__<<":"<<__LINE__<<":"<<*ptrME0REUIE2CEU<<std::endl;

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
