# License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/DFKI-Interactive-Machine-Learning/ophthalmo-cdss">Ophthalmo-CDSS</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/robertleist">Robert Andreas Leist</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>

# Tests
This folder contains the tests for the dashboard. The tests are implemented using the `unittest` framework and assure 
that the functionality of the dashboard is working as expected. Testing the dashboard is an extensive task and the tests
are not exhaustive. You can run the tests by executing the following command in the root directory of the repository:
```bash
python -m unittest tests/tests.py
```
Note that tests require multiple reruns of the dashboard, which can be very time consuming!