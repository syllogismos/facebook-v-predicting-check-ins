import requests

url = "https://www.kaggle.com/competitions/submissions/accept"

payload = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"CompetitionId\"\r\n\r\n5186\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"IsScriptVersionSubmission\"\r\n\r\nFalse\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"SubmissionDescription\"\r\n\r\ntesting submission post request\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"__RequestVerificationToken\"\r\n\r\nmx3QP5qrt0Gy0_MW1NeFRkRN180Cmexph5GvafxJSyW884duL4482-rGF-tJzZw2wW2PDvpvSxF9G9io9NwrcJgR41U1\r\n-----011000010111000001101001--"
headers = {
    'origin': "https://www.kaggle.com",
    'x-devtools-emulate-network-conditions-client-id': "54735B3F-4990-4D62-AAC6-05DFAD625CF6",
    'upgrade-insecure-requests': "1",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36",
    'content-type': "multipart/form-data; boundary=---011000010111000001101001",
    'accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    'dnt': "1",
    'referer': "https://www.kaggle.com/c/facebook-v-predicting-check-ins/submissions/attach",
    'accept-encoding': "gzip, deflate, br",
    'accept-language': "en-US,en;q=0.8,ms;q=0.6",
    'cookie': "optimizelyEndUserId=oeu1463206818614r0.999748940163294; __insp_wid=22567980; __insp_slim=1465708180779; __insp_nv=true; __insp_ref=d; __insp_targlpu=https%3A%2F%2Fwww.kaggle.com%2F; __insp_targlpt=Kaggle%3A%20Your%20Home%20for%20Data%20Science; __insp_norec_sess=true; __RequestVerificationToken=dVwexvnPeKLtBVSlSkf62DYU57zm9T6m0i3oBKu-j-cUcmUZPgwi-6XfJrUv4E3xP1WUkSXgx_EjQ_VxbcGx4AEFlMo1; .ASPXAUTH=486EE5890B1841237F11706E23C65EE3473EEA731D79BFABB1EB3D287E7A08C2176D37E6DB5445BCDBB973C7C50E3BDC8C531FDECDB45831BEBB3BABDC5EE092CEFC376A7B2BF39C23DD6623EF2FBA6946498E79; optimizelySegments=%7B%225684981785%22%3A%22false%22%2C%225646609362%22%3A%22search%22%2C%225668443100%22%3A%22gc%22%7D; optimizelyBuckets=%7B%7D; ARRAffinity=0227a09cff6fb7f015e77569c300a7d28b5f79ea601319ffe879df956d9fab88; __utmt=1; __utma=158690720.242139384.1463206821.1467060699.1467106839.47; __utmb=158690720.12.10.1467106839; __utmc=158690720; __utmz=158690720.1467006991.45.6.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided)",
    'cache-control': "no-cache",
    'postman-token': "404c3b96-a2b9-9d5a-40c8-d55c7d9a3a70"
    }

response = requests.request("POST", url, data=payload, headers=headers)

print(response.text)