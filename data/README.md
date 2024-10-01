# Naming scheme
"(CPU name)_(target CPU class).txt" 

Further information on the 'x86-64-v*' targets can be found at https://developers.redhat.com/blog/2021/01/05/building-red-hat-enterprise-linux-9-for-the-x86-64-v2-microarchitecture-level#background_of_the_x86_64_microarchitecture_levels

# Parsing
`./scripts/plot_utils.py` demonstrates how to parse this, and can be used as a python library.
It's not an efficient parser, but there isn't that much data.
