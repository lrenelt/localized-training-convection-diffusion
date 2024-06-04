add_library(ipyultraweak SHARED pyultraweak-interactive.cc)
target_link_libraries(ipyultraweak ${PYTHON_LIBRARIES})
set_target_properties(ipyultraweak PROPERTIES PREFIX "")
