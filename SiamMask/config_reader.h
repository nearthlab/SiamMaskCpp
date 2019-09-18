#include <dlib/config_reader.h>

namespace dlib {

    template<typename T>
    T cast_from_string(const std::string& str) {
        std::istringstream sin(str);
        T value;
        sin >> value;
        return value;
    }

    template<typename T>
    std::ostream& operator << (std::ostream& out, const std::vector<T>& v) {
        out << "[";
        for(unsigned long i = 0; i < v.size(); ++i) {
            if (i > 0)
                out << ", ";
            out << v[i];
        }
        out << "]";

        return out;
    }

    void remove_space(std::string& str) {
        str.erase(
            std::remove_if(
                str.begin(), str.end(),
                [](unsigned char x){ return std::isspace(x); }
            ),
            str.end()
        );
    }

    void strip_brackets(std::string& str) {
        auto first_bracket = str.find_first_of('[');
        if(first_bracket == std::string::npos) {
            std::ostringstream sout;
            sout << "Could not find a left bracket in " << str;
            throw std::runtime_error(sout.str());
        }
        str.erase(str.begin() + first_bracket);

        auto last_bracket = str.find_last_of(']');
        if(last_bracket == std::string::npos) {
            std::ostringstream sout;
            sout << "Could not find a right bracket in " << str;
            throw std::runtime_error(sout.str());
        }
        str.erase(str.begin() + last_bracket);
    }

    template<typename T>
    std::istream& operator >> (std::istream& in, std::vector<T>& v) {
        v.clear();

        std::string str;
        std::getline(in, str, '\n');

        if(str.empty()) return in;
        remove_space(str);
        strip_brackets(str);

        std::istringstream sin(str);
        while(sin.good())
        {
            std::string substr;
            std::getline(sin, substr, ',');
            if(!substr.empty()) v.push_back(cast_from_string<T>(substr));
        }

        return in;
    }

    // Add vector I/O compatible with argparse's
    template <typename cr_type, typename T>
    typename enable_if<is_config_reader<cr_type>, std::vector<T> >::type get_option (
        const cr_type& cr,
        const std::string& option_name,
        const std::vector<T>& default_value
    ) {
        std::string value_str = get_option(cr, option_name, cast_to_string(default_value));
        return cast_from_string<std::vector<T> >(value_str);
    }
}
