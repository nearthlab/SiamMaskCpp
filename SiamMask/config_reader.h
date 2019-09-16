#include <dlib/config_reader.h>
#include <argparse/argparse.hpp>

namespace dlib {
    // Add vector I/O compatible with argparse's
    template <typename cr_type, typename T>
    typename enable_if<is_config_reader<cr_type>, std::vector<T> >::type get_option (
        const cr_type& cr,
        const std::string& option_name,
        const std::vector<T>& default_value
    ) {
        std::string value_str = get_option(cr, option_name, argparse::toString(default_value));
        return argparse::castTo<std::vector<T> >(value_str);
    }
}
