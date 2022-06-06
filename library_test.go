package main

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import (
	"bufio"
	"strings"
	"testing"
)

func TestOboParsing(*testing.T) {
	var obolist []*OboTermEntry

	parentchildren := make(map[string][]*OboTermEntry)
	obochan := make(chan *OboTermEntry)

	s := `

format-version: 1.2
date: 17:11:2011 13:07
saved-by: lschriml
auto-generated-by: OBO-Edit 2.1-beta6
default-namespace: symptoms

[Typedef]
id: part_of
name: part_of

[Term]
id: SYMP:0000000
! We do not care about this comment
name: cellulitis
def: "Cellulitis is a musculoskeletal system symptom characterized as a diffuse and especially subcutaneous inflammation of connective tissue." [URL:http\://www2.merriam-webster.com/cgi-bin/mwmednlm?book=Medical&va=cellulitis]
is_a: SYMP:0000891 ! musculoskeletal system symptom

[Term]
id: SYMP:0000001
name: abdominal cramp
is_a: SYMP:0000461 ! abdominal symptom

[Term]
id: SYMP:0000002
name: abdominal distention
is_a: SYMP:0000461 ! abdominal symptom

[Term]
id: SYMP:0000003
name: acute enteritis in newborns
is_obsolete: true

[Term]
id: SYMP:0000004
name: arrested moulting
is_obsolete: true
[Term]
id: SYMP:0000005
name: ataxia
def: "Ataxia is a neurological and physiological symptom characterized by an inability to coordinate voluntary muscular movements that is symptomatic of some nervous disorders." [URL:http\://www2.merriam-webster.com/cgi-bin/mwmednlm?book=Medical&va=ataxia]
synonym: "uncoordination" EXACT []
is_a: SYMP:0000410 ! neurological and physiological symptom

[Term]
id: SYMP:0000006
name: backache
def: "Backache is a pain occurring in the lower back." [URL:http\://www2.merriam-webster.com/cgi-bin/mwmednlm?book=Medical&va=backache]
synonym: "back pain" EXACT []
is_a: SYMP:0000099 ! pain

[Term]
id: SYMP:0000007
name: bleeding
def: "A general symptom that is characterized as an act, instance, or result of being bled or the process by which something is bled: as a the escape of blood from vessels." [url:http\://www.merriam-webster.com/medlineplus/bleeding]
is_a: SYMP:0000567 ! general symptom

[Term]
id: SYMP:0000008
name: blindness
is_a: SYMP:0000320 ! vision symptom

[Term]
id: SYMP:0000009
name: blister
def: "Blister is a skin and integumentary tissue symptom characterized as a fluid-filled elevation of the epidermis." [url:http\://www2.merriam-webster.com/cgi-bin/mwmednlm?book=Medical&va=blister]`

	stringreader1 := bufio.NewReader(strings.NewReader(s))
	stringreader2 := bufio.NewReader(strings.NewReader(s))

	obolist, parentchildren = ParseToSlice(*stringreader1, parentchildren, obolist)
	obochan = ParseToChannel(*stringreader2, obochan)

	for ent := range obochan {
		obolist = append(obolist, ent)
	}

	Dump(obolist, parentchildren)
}
