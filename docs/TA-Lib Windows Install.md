# How to Properly Install TA-Lib on Windows 11 for Python: A Step-by-Step Guide | by Arjun Krishna K | Python’s Gurus | Medium

# How to Properly Install TA-Lib on Windows 11 for Python: A Step-by-Step Guide

[

![Arjun Krishna K](https://miro.medium.com/v2/da:true/resize:fill:64:64/0*prgUkxJbEEE8mp2s)





](/@arjunkrish?source=post_page---byline--13ebb684f4a6---------------------------------------)

[Arjun Krishna K](/@arjunkrish?source=post_page---byline--13ebb684f4a6---------------------------------------)

Follow

3 min read

·

May 29, 2024

[

](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fvote%2Fpythons-gurus%2F13ebb684f4a6&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fpythons-gurus%2Fhow-to-properly-install-ta-lib-on-windows-11-for-python-a-step-by-step-guide-13ebb684f4a6&user=Arjun+Krishna+K&userId=278c6daae0c0&source=---header_actions--13ebb684f4a6---------------------clap_footer------------------)

207

5

[](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fbookmark%2Fp%2F13ebb684f4a6&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fpythons-gurus%2Fhow-to-properly-install-ta-lib-on-windows-11-for-python-a-step-by-step-guide-13ebb684f4a6&source=---header_actions--13ebb684f4a6---------------------bookmark_footer------------------)

[

Listen









](/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3D13ebb684f4a6&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fpythons-gurus%2Fhow-to-properly-install-ta-lib-on-windows-11-for-python-a-step-by-step-guide-13ebb684f4a6&source=---header_actions--13ebb684f4a6---------------------post_audio_button------------------)

Share

TA-Lib (Technical Analysis Library) is a used library for the technical analysis of financial data, containing over 150 functions for analyzing stock market data. This guide will take you through the detailed steps to install the TA-Lib C library and its Python wrapper on a Windows system.

![](https://miro.medium.com/v2/resize:fit:1050/0*PdSmdtXYiNUevbpM)
Photo by rivage on Unsplash

**Step 1: Ensure TA-Lib C Library is Properly Installed**

1\. Go to the TA-Lib Download Page

[**_Ta-Lib_ Home page Link**](https://sourceforge.net/projects/ta-lib/)

2\. Download the source code on the home page by clicking Download

![](https://miro.medium.com/v2/resize:fit:1050/1*UhPBsmMmupGdAfvT4Kj6OQ.png)

3\. Extract the Source Code: Extract the downloaded archive to a directory, for example, `C:\ta-lib`.

![](https://miro.medium.com/v2/resize:fit:953/1*zsiXkIVFl-8TF0akamFDKA.png)

4\. Build the TA-Lib C Library

1.  Open Visual Studio installer and click modify

![](https://miro.medium.com/v2/resize:fit:1050/1*0Z7HfnekakFm4ZJ81Z0HNg.png)

2\. Ensure that the following workloads(Desktop development with C++)and individual components are installed. If they are not already installed, please select and install them

![](https://miro.medium.com/v2/resize:fit:1050/1*DwORtc7s1gPtjAVYpYrVrg.png)

3\. Open the following command prompt

![](https://miro.medium.com/v2/resize:fit:1050/1*DkLdjJAB0mLQ_0G0pdryVw.png)

4\. Navigate to the Directory

cd C:\\ta-lib\\c\\make\\cdr\\win32\\msvc

5\. Run the Build Command

nmake

6\. Set the TA\_INCLUDE\_PATH and TA\_LIBRARY\_PATH Environment Variables

set TA\_INCLUDE\_PATH=C:\\ta-lib\\c\\include  
set TA\_LIBRARY\_PATH=C:\\ta-lib\\c\\lib

**Step 3: Manually Install the Python Wrapper with Specified Paths**

## Get Arjun Krishna K’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

Subscribe

1\. Activate Your Virtual Environment (if not already activated)

yourpath\\venv\\Scripts\\activate

2\. Clone the TA-Lib Python Wrapper Repository

git clone https://github.com/mrjbq7/ta-lib.git   
cd ta-lib

3\. Build and Install the Python Wrapper

python setup.py build\_ext --include-dirs=C:\\ta-lib\\c\\include --library-dirs=C:\\ta-lib\\c\\lib  
python setup.py install

Step 4: Verify Installation

import talib  
print(len(talib.get\_functions()))  
Output : 158

## Python’s Gurus🚀

_Thank you for being a part of the_ [**_Python’s Gurus community_**](https://medium.com/pythons-gurus)**_!_**

_Before you go:_

-   Be sure to **clap x50 time** and **follow** the writer ️👏**️️**
-   Follow us: [**Newsletter**](/pythons-gurus/newsletters/gurus-advisor)
-   Do you aspire to become a Guru too? **_Submit your best article or draft_** to reach our audience.

## Embedded Content

---