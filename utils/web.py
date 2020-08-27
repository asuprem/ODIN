import sys, requests

# from https://sumit-ghosh.com/articles/python-download-progress-bar/ 
def download(file_, url):
    sys.stdout.write("Downloading %s from %s"%(file_, url))
    with open(file_, 'wb') as f:
      response = requests.get(url, stream=True)
      all_data = response.headers.get('content-length')
      if all_data is None:
        f.write(response.content)
      else:
        all_data = int(all_data)
        progress = 0
        for chunk in response.iter_content(chunk_size=max(int(all_data/1000), 1024*1024)):
          progress += len(chunk)
          f.write(chunk)
          sys.stdout.write('\r{}/{} bytes [{}{}]'.format(all_data,progress,'█' * int(100*progress/all_data), '.' * (100-int(100*progress/all_data))))
          sys.stdout.flush()
    sys.stdout.write('\nDownload of %s to %s completed\n'%(file_, url))