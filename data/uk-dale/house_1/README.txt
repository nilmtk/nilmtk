
This dataset was recorded from a London end-of-terrace Victorian house,
built around 1905, with lots of insulation added recently.


DATA FORMAT
===========

Please see:
https://github.com/JackKelly/rfm_ecomanager_logger/wiki/Data-format


SENSORS USED
============

For details of these sensors, see:
https://github.com/JackKelly/rfm_ecomanager_logger/wiki/Data-format#wiki-Two_types_of_sensor


Current Transformer (CT) sensors (recording apparent power in units of VA)
--------------------------------------------------------------------------

"aggregate", "boiler", "solar_thermal", "kitchen_lights" and "lighting_circuit"



Individual Appliance Monitors (recording real power in units of watts)
----------------------------------------------------------------------

All other channels.


NOTEWORTHY APPLICANES
=====================

soldering_iron
 this is a temperature controlled soldering iron, a Xytronic 168-3CD.

boiler
 includes all electronics associated with the boiler including the
 central heating pump, the hot water pump, the bathroom underfloor heating
 pump, the boiler controller, the boiler itself.
 Over winter the central heating is on 24 hrs and is controlled by our
 portable wireless thermostat which is usually set at 18-20 degrees C
 and is put in the room we want to be the most comfortable.
 Prior to 3rd May 2013, the hot water was set to come on from 0630-0700
 and 1630-1700.  After 3rd May the HW comes on 0650-0700 and 1650-1700.

solar_thermal
 includes all electronics associated with the evacuated-tube solar
 hot water system including the water pump and control electronics.

tv
 Panasonic 34" widescreen CRT TV in the livingroom.  There is only one
 TV in the house.

kitchen_lights
 10 x 50watt, 12volt tungsten lamps. All on a 230 volt TRIAC dimmer.
 The kitchen receives very little natural light hence the kitchen lights
 are used a lot.

 5th April 2013 1450 BST:
   replaced 1x50W halogen with 10W 12V Philips dimmable LED

 10th April 2013:
   replaced 1x50W halogen with 8W 12V MegaMan Dimmable LED

 25th April 2013 0800 BST:
   all 10 light fittings are now 10W 12V Philips dimmable LEDs

htpc
 home theatre PC. The only AV source for the TV.
 Also turns itself on to record FreeView programs. Also used for
 playing music occasionally.

fridge
 combined fridge & freezer, bought around 2010

gigE_switch
 8-port gigabit Ethernet switch in the office.  Turned off when not in use.

utilityrm_lamp
 A fluorescent lamp in the utility room (where the washing machine is).

bedroom_chargers
 including: iPhone 4s & cordless phone & baby monitor &
 bedroom DAB radio charger (the DAB radio has a built-in battery so this
 charger probably has three power levels:
    radio on (battery fully charged)
    charge battery (radio off)
    charger battery and run radio

childs_ds_lamp
 Prior to around 1st April 2013 it was a dimmable CFL.  But that blew so we changed
 to a 75W incandesent for a little while.  Then on 10th April 2013 we changed it
 to a Philips MASTER LEDBULB 8W dimmable.

data_logging_pc
 This meter sometimes has a large external USB hard disk plugged in at
 the same time as the PC.

Some sensors are switched off from the socket for significant portions
 of time.  These include: laptop, lcd_office, hifi_office,
 soldering_iron, gigE_switch, hoover, utilityrm_lamp, hair_dry,
 straighteners, iron

NOTES ABOUT KETTLE AND TOASTER CHANNELS
=======================================

Both the Kettle and Toaster channels are sometimes used for other
appliances:

Kettle:
 * Kettle
 * Breville Food mixer (for milkshakes)
 * toaster sandwich maker

Toaster: 
 * Toaster
 * Artisan kitchen aid
 * Kenwood food mixer

For the five days from Mon 24th June 2013 to Fri 28th June we had
someone staying at the house who occassionally swapped the toaster and
kettle around (i.e. the toaster was plugged into the kettle sensor and
visa-versa!) and also appeared to plug the hoover sensor into the
kettle sensor (i.e. both the hoover and kettle sensor would have
recorded the same appliance for a few hours).


APPLIANCES NOT RECORDED INDIVIDUALLY
====================================

Immersion heater
  It has never been used and would only ever be used if the boiler broke.

Living room underfloor heating pump (very efficient)
  Only uses 5 watts (it has a power consumption display) and would be hard
  to attach a sensor to.  In winter months this is turned on around 7:30am and
  is left on until 10:30pm.

Burglar alarm
  Always on.  Appears to use about 10 watts.  Was turned off Sunday
  11th August 2013.

Bathroom extractor fan (MVHR)
  Always on during winter months (in summer we turn the fan off and open the
  window). Has 2 modes: trickle and boost.  Boost is triggered using a manual
  pull-cord when necessary. Only uses about 2 watts in trickle mode and about
  10 watts in boost mode.

Power drill
  Used:
    * Sat 13/04/2013 17:43 BST for one short burst

Dell laptop
  Charged 09:21 BST Sat 4th May 2013


ABBREVIATIONS USED FOR LAMP NAMES
=================================

Example lamp appliance names are "livingroom_s_lamp" or "bedroom_ds_lamp".
The abbreviations used are:

 d = "dimmable"
 s = "standing lamp"
 t = "table lamp"


LIGHTING CIRCUIT INCLUDES KITCHEN LIGHTS
========================================

The "lighting_circuit" channel was recorded at the consumer box and includes ALL
ceiling lights, including the kitchen lights.  Hence any power used by the
kitchen ceiling lights will also register on the "lighting_circuit" channel.
