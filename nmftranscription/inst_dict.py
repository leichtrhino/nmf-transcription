
melodic_dict = {
    'Acoustic Piano': (1, 'Piano'),
    'Bright Piano': (2, 'Piano'),
    'Electric Grand Piano': (3, 'Piano'),
    'Honky-tonk Piano': (4, 'Piano'),
    'Electric Piano': (5, 'Piano'),
    'Electric Piano 2': (6, 'Piano'),
    'Harpsichord': (7, 'Piano'),
    'Clavi': (8, 'Piano'),
    'Celesta': (9, 'Chromatic Percussion'),
    'Glockenspiel': (10, 'Chromatic Percussion'),
    'Musical box': (11, 'Chromatic Percussion'),
    'Vibraphone': (12, 'Chromatic Percussion'),
    'Marimba': (13, 'Chromatic Percussion'),
    'Xylophone': (14, 'Chromatic Percussion'),
    'Tubular Bell': (15, 'Chromatic Percussion'),
    'Dulcimer': (16, 'Chromatic Percussion'),
    'Drawbar Organ': (17, 'Organ'),
    'Percussive Organ': (18, 'Organ'),
    'Rock Organ': (19, 'Organ'),
    'Church organ': (20, 'Organ'),
    'Reed organ': (21, 'Organ'),
    'Accordion': (22, 'Organ'),
    'Harmonica': (23, 'Organ'),
    'Tango Accordion': (24, 'Organ'),
    'Acoustic Guitar (nylon)': (25, 'Guitar'),
    'Acoustic Guitar (steel)': (26, 'Guitar'),
    'Electric Guitar (jazz)': (27, 'Guitar'),
    'Electric Guitar (clean)': (28, 'Guitar'),
    'Electric Guitar (muted)': (29, 'Guitar'),
    'Overdriven Guitar': (30, 'Guitar'),
    'Distortion Guitar': (31, 'Guitar'),
    'Guitar harmonics': (32, 'Guitar'),
    'Acoustic Bass': (33, 'Bass'),
    'Electric Bass (finger)': (34, 'Bass'),
    'Electric Bass (pick)': (35, 'Bass'),
    'Fretless Bass': (36, 'Bass'),
    'Slap Bass 1': (37, 'Bass'),
    'Slap Bass 2': (38, 'Bass'),
    'Synth Bass 1': (39, 'Bass'),
    'Synth Bass 2': (40, 'Bass'),
    'Violin': (41, 'Strings'),
    'Viola': (42, 'Strings'),
    'Cello': (43, 'Strings'),
    'Double bass': (44, 'Strings'),
    'Tremolo Strings': (45, 'Strings'),
    'Pizzicato Strings': (46, 'Strings'),
    'Orchestral Harp': (47, 'Strings'),
    'Timpani': (48, 'Strings'),
    'String Ensemble 1': (49, 'Ensemble'),
    'String Ensemble 2': (50, 'Ensemble'),
    'Synth Strings 1': (51, 'Ensemble'),
    'Synth Strings 2': (52, 'Ensemble'),
    'Voice Aahs': (53, 'Ensemble'),
    'Voice Oohs': (54, 'Ensemble'),
    'Synth Voice': (55, 'Ensemble'),
    'Orchestra Hit': (56, 'Ensemble'),
    'Trumpet': (57, 'Brass'),
    'Trombone': (58, 'Brass'),
    'Tuba': (59, 'Brass'),
    'Muted Trumpet': (60, 'Brass'),
    'French horn': (61, 'Brass'),
    'Brass Section': (62, 'Brass'),
    'Synth Brass 1': (63, 'Brass'),
    'Synth Brass 2': (64, 'Brass'),
    'Soprano Sax': (65, 'Reed'),
    'Alto Sax': (66, 'Reed'),
    'Tenor Sax': (67, 'Reed'),
    'Baritone Sax': (68, 'Reed'),
    'Oboe': (69, 'Reed'),
    'English Horn': (70, 'Reed'),
    'Bassoon': (71, 'Reed'),
    'Clarinet': (72, 'Reed'),
    'Piccolo': (73, 'Pipe'),
    'Flute': (74, 'Pipe'),
    'Recorder': (75, 'Pipe'),
    'Pan Flute': (76, 'Pipe'),
    'Blown Bottle': (77, 'Pipe'),
    'Shakuhachi': (78, 'Pipe'),
    'Whistle': (79, 'Pipe'),
    'Ocarina': (80, 'Pipe'),
    'Lead 1 (square)': (81, 'Synth Lead'),
    'Lead 2 (sawtooth)': (82, 'Synth Lead'),
    'Lead 3 (calliope)': (83, 'Synth Lead'),
    'Lead 4 (chiff)': (84, 'Synth Lead'),
    'Lead 5 (charang)': (85, 'Synth Lead'),
    'Lead 6 (voice)': (86, 'Synth Lead'),
    'Lead 7 (fifths)': (87, 'Synth Lead'),
    'Lead 8 (bass + lead)': (88, 'Synth Lead'),
    'Pad 1 (Fantasia)': (89, 'Synth Pad'),
    'Pad 2 (warm)': (90, 'Synth Pad'),
    'Pad 3 (polysynth)': (91, 'Synth Pad'),
    'Pad 4 (choir)': (92, 'Synth Pad'),
    'Pad 5 (bowed)': (93, 'Synth Pad'),
    'Pad 6 (metallic)': (94, 'Synth Pad'),
    'Pad 7 (halo)': (95, 'Synth Pad'),
    'Pad 8 (sweep)': (96, 'Synth Pad'),
    'FX 1 (rain)': (97, 'Synth Effects'),
    'FX 2 (soundtrack)': (98, 'Synth Effects'),
    'FX 3 (crystal)': (99, 'Synth Effects'),
    'FX 4 (atmosphere)': (100, 'Synth Effects'),
    'FX 5 (brightness)': (101, 'Synth Effects'),
    'FX 6 (goblins)': (102, 'Synth Effects'),
    'FX 7 (echoes)': (103, 'Synth Effects'),
    'FX 8 (sci-fi)': (104, 'Synth Effects'),
    'Sitar': (105, 'Ethnic'),
    'Banjo': (106, 'Ethnic'),
    'Shamisen': (107, 'Ethnic'),
    'Koto': (108, 'Ethnic'),
    'Kalimba': (109, 'Ethnic'),
    'Bagpipe': (110, 'Ethnic'),
    'Fiddle': (111, 'Ethnic'),
    'Shanai': (112, 'Ethnic'),
    'Tinkle Bell': (113, 'Percussive'),
    'Agogo': (114, 'Percussive'),
    'Steel Drums': (115, 'Percussive'),
    'Woodblock': (116, 'Percussive'),
    'Taiko Drum': (117, 'Percussive'),
    'Melodic Tom': (118, 'Percussive'),
    'Synth Drum': (119, 'Percussive'),
    'Reverse Cymbal': (120, 'Percussive'),
    'Guitar Fret Noise': (121, 'Sound effects'),
    'Breath Noise': (122, 'Sound effects'),
    'Seashore': (123, 'Sound effects'),
    'Bird Tweet': (124, 'Sound effects'),
    'Telephone Ring': (125, 'Sound effects'),
    'Helicopter': (126, 'Sound effects'),
    'Applause': (127, 'Sound effects'),
    'Gunshot': (128, 'Sound effects'),
}

percussion_dict = {
    'Bass Drum 2': (35, 'Percussion'),
    'Bass Drum 1': (36, 'Percussion'),
    'Side Stick': (37, 'Percussion'),
    'Snare Drum 1': (38, 'Percussion'),
    'Hand Clap': (39, 'Percussion'),
    'Snare Drum 2': (40, 'Percussion'),
    'Low Tom 2': (41, 'Percussion'),
    'Closed Hi-hat': (42, 'Percussion'),
    'Low Tom 1': (43, 'Percussion'),
    'Pedal Hi-hat': (44, 'Percussion'),
    'Mid Tom 2': (45, 'Percussion'),
    'Open Hi-hat': (46, 'Percussion'),
    'Mid Tom 1': (47, 'Percussion'),
    'High Tom 2': (48, 'Percussion'),
    'Crash Cymbal 1': (49, 'Percussion'),
    'High Tom 1': (50, 'Percussion'),
    'Ride Cymbal 1': (51, 'Percussion'),
    'Chinese Cymbal': (52, 'Percussion'),
    'Ride Bell': (53, 'Percussion'),
    'Tambourine': (54, 'Percussion'),
    'Splash Cymbal': (55, 'Percussion'),
    'Cowbell': (56, 'Percussion'),
    'Crash Cymbal 2': (57, 'Percussion'),
    'Vibra Slap': (58, 'Percussion'),
    'Ride Cymbal 2': (59, 'Percussion'),
    'High Bongo': (60, 'Percussion'),
    'Low Bongo': (61, 'Percussion'),
    'Mute High Conga': (62, 'Percussion'),
    'Open High Conga': (63, 'Percussion'),
    'Low Conga': (64, 'Percussion'),
    'High Timbale': (65, 'Percussion'),
    'Low Timbale': (66, 'Percussion'),
    'High Agogo': (67, 'Percussion'),
    'Low Agogo': (68, 'Percussion'),
    'Cabasa': (69, 'Percussion'),
    'Maracas': (70, 'Percussion'),
    'Short Whistle': (71, 'Percussion'),
    'Long Whistle': (72, 'Percussion'),
    'Short Guiro': (73, 'Percussion'),
    'Long Guiro': (74, 'Percussion'),
    'Claves': (75, 'Percussion'),
    'High Wood Block': (76, 'Percussion'),
    'Low Wood Block': (77, 'Percussion'),
    'Mute Cuica': (78, 'Percussion'),
    'Open Cuica': (79, 'Percussion'),
    'Mute Triangle': (80, 'Percussion'),
    'Open Triangle': (81, 'Percussion'),
}
