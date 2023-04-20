#!/bin/bash
make -j8 mpir&&
make Fake-Offline.x highgear-party.x&&
sudo make -j 8 online&&
./Fake-Offline.x 3 -lgp 128&&
./Fake-Offline.x 4 -lgp 128&&
./Fake-Offline.x 5 -lgp 128&&
./Fake-Offline.x 6 -lgp 128
