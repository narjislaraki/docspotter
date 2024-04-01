from distutils.core import setup
setup(
  name = 'docspotter',        
  packages = ['docspotter'],  
  version = '0.3',      
  license='MIT',        
  description = 'DocSpotter is a Python library designed to extract specific information from document images by combining text detection and extraction technologies.',  
  author = 'Nlaraki',                   
  author_email = 'larakinarjis@gmail.com',     
  url = 'https://github.com/user/reponame',   
  download_url = 'https://github.com/narjislaraki/docspotter',   
  keywords = ['OCR', 'CRAFT', 'text', 'detection', 'extraction', 'easy'],   
  install_requires=[           
          'pytesseract',
          'opencv-python',
          'craft-text-detector',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.9',

  ],
)