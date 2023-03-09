#include "ProgramSection.hpp"

#include <chrono>
#include <string>

std::string 
ProgramSection::to_string() {

  std::chrono::duration<double> seconds = end - start;

  std::string tmp = time_ticks < 0.000001 ? (name + " --- " + std::to_string(seconds.count()) + " s") : (name + " --- " + std::to_string(time_ticks) + " s");

  for (size_t i = 0; i < subsections.size(); i++)
  {
    tmp = tmp.append("\n|\n").append(subsections[i].to_string());
  }

  return tmp;
}

ProgramSection
ProgramSection::init_program_structure(std::string program_name) {
  ProgramSection ps(program_name);
  
  if (program_name == "RADMeth_regression"){
    ps.add_subsection(ProgramSection("+--- Parameter validation"));
    ps.add_subsection(ProgramSection("+--- Main loop"));
  }
  
  return ps;
}