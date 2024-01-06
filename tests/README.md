# Test Suite
![tests](https://gist.githubusercontent.com/eangius/eb12b64cf81f991888c6bfd3f3419064/raw/tests_badge.svg)
![coverage](https://gist.githubusercontent.com/eangius/eb12b64cf81f991888c6bfd3f3419064/raw/coverage_badge.svg)

You can run various subsets of the tests as follows:
```bash
./run --test;                                    # everything
./run --test -m <test_type>;                     # unit, integration or stress
./run --test tests/<test_file.py>::<test_func>;  # specific tests
./run --test -k '[<test_name_pattern>]';         # parametrized auto generated
```
