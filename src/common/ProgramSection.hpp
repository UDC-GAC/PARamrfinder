#include <chrono>
#include <string>
#include <vector>
#include <limits>

class ProgramSection {
public:
    std::vector<ProgramSection> subsections;

    ProgramSection(std::string nm) {
        name = nm;
        time_ticks = 0.0;
        start = std::chrono::high_resolution_clock::now();
        end = std::chrono::high_resolution_clock::now();
    }

    void
    start_section() {
        start = std::chrono::high_resolution_clock::now();
    }

    void
    end_section() {
        end = std::chrono::high_resolution_clock::now();
    }

    void
    store_time() {
      time_ticks += std::chrono::duration<double>(end - start).count();
    }

    double
    get_elapsed_time() {
         return time_ticks < 0.000001 ? 
           std::chrono::duration<double>(end - start).count() : 
           time_ticks;
    }

    void
    set_elapsed_time(double time) {
       time_ticks = time;
    }

    void
    add_subsection(ProgramSection ps) {
        subsections.push_back(ps);
    }

    std::string 
    to_string();

    static ProgramSection init_program_structure(std::string program_name);

private:
    std::string name;
    double time_ticks;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
};