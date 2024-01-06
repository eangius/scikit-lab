# Test Suite

![Tests](https://gist.githubusercontent.com/eangius/eb12b64cf81f991888c6bfd3f3419064/raw/80e7064c1af6631c8a7349cb6ab253106a415800/tests_badge.svg)
![Coverage](https://gist.githubusercontent.com/eangius/eb12b64cf81f991888c6bfd3f3419064/raw/cc3cd9ef3be305d0006072dd76b8085ae7925e7e/coverage_badge.svg)

You can run various subsets of the tests as follows:
```bash
./run --test;                                    # everything
./run --test -m <test_type>;                     # unit, integration or stress
./run --test tests/<test_file.py>::<test_func>;  # specific tests
./run --test -k '[<test_name_pattern>]';         # parametrized auto generated  
```
