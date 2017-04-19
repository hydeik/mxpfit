#ifndef EXAMPLES_TIMER_HPP
#define EXAMPLES_TIMER_HPP

#include <chrono>
#include <type_traits>

//
// A timer class for measuring elapsed time.
//
// This class has similar interface to boost::timer, uses std::chrono
// internally.
//
class Timer
{
public:
    using clock_type =
        typename std::conditional<std::chrono::high_resolution_clock::is_steady,
                                  std::chrono::high_resolution_clock,
                                  std::chrono::steady_clock>::type;
    using microseconds = std::chrono::microseconds;

    Timer() : m_start_time(clock_type::now())
    {
    }

    Timer(const Timer&) = default;

    ~Timer() = default;

    void restart()
    {
        m_start_time = clock_type::now();
    }

    microseconds elapsed() const
    {
        return std::chrono::duration_cast<microseconds>(clock_type::now() -
                                                        m_start_time);
    }

private:
    clock_type::time_point m_start_time;
};

#endif /* EXAMPLES_TIMER_HPP */
