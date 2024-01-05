# Test Suite

[//]: [![Tests](../runtime/code_tests/badge.svg)]()
[//]: [![Coverage](../runtime/code_coverage/badge.svg)](../runtime/code_coverage/index.html)

We can run various subsets of the tests as follows:
```bash
./run --test;                                    # everything
./run --test -m <test_type>;                     # unit, integration or stress
./run --test tests/<test_file.py>::<test_func>;  # specific tests
./run --test -k '[<test_name_pattern>]';         # parametrized auto generated  
```
