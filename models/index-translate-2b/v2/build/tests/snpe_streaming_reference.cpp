// CPU-only regression: partition residency, recurrent state, prefill, and vocabulary assembly.
#include "vendor_decoder.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
extern "C" VendorDecoder* ocrCreateSnpe(const char*,const char*,bool);
using json=nlohmann::json;
void equal(const std::vector<float>& a,const std::vector<float>& b) {
 if(a.size()!=b.size())throw std::runtime_error("logit size mismatch");
 for(size_t i=0;i<a.size();++i)if(!std::isfinite(a[i]) || std::abs(a[i]-b[i])>1e-4f)
  throw std::runtime_error("stream/prefill/shard numerical mismatch at "+std::to_string(i));
}
int main(int argc,char**argv) {
 try {
  if(argc!=2)throw std::runtime_error("Usage: snpe_streaming_reference fixture_directory");
  std::string folder=argv[1];json full,sharded;
  std::ifstream(folder+"/full.json")>>full;std::ifstream(folder+"/sharded.json")>>sharded;
  auto resident=std::unique_ptr<VendorDecoder>(ocrCreateSnpe(folder.c_str(),full.dump().c_str(),true));
  sharded["testStreamPartitions"]=true;
  auto stream=std::unique_ptr<VendorDecoder>(ocrCreateSnpe(folder.c_str(),sharded.dump().c_str(),true));
  const std::vector<int> prompt={3,5,7,11};
  for(int repeat=0;repeat<3;++repeat) {
   resident->reset();stream->reset();std::vector<float> expected;
   for(size_t i=0;i<prompt.size();++i)expected=resident->step(prompt[i],i);
   equal(expected,stream->prefill(prompt));
   equal(resident->step(13,4),stream->step(13,4));
   equal(resident->step(17,5),stream->step(17,5));
  }
  std::atomic_bool cancelled{true};bool stopped=false;
  try{stream->prefill(prompt,&cancelled);}catch(const std::runtime_error& e){stopped=std::string(e.what()).find("cancelled")!=std::string::npos;}
  if(!stopped)throw std::runtime_error("cancellation ignored");
  for(int position:{-1,256,257}) {
   bool rejected=false;try{stream->step(0,position);}catch(const std::runtime_error&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid position accepted");
  }
  sharded["decoder"]["partitions"].back()["vocabOffset"]=511;
  bool rejected=false;
  try{std::unique_ptr<VendorDecoder> bad(ocrCreateSnpe(folder.c_str(),sharded.dump().c_str(),true));}
  catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("overlapping projection accepted");
  std::cout<<"PASS: CPU reference only; streamed state + prefill + sharded logits + reset + cancellation + bounds"<<std::endl;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
