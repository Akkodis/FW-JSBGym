#!/usr/bin/env python3
import jsbsim
import os
import time

FG_OUT_FILE = 'flightgear.xml'
fdm = jsbsim.FGFDMExec(None)
fdm.set_output_directive(os.path.join(os.path.dirname(os.path.abspath(__file__)), FG_OUT_FILE))
fdm.load_script('scripts/c1723.xml')
fdm.run_ic()
# fdm.enable_output()

while fdm.run():
    time.sleep(0.01)
    pass