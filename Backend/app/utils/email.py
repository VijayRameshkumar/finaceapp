import msal
import requests

client_id = 'xxx'
tenant_id = 'xxx'
client_secret = 'xxx'
user_email = 'xxx'

authority_url = f'https://login.microsoftonline.com/{tenant_id}'
scope = ['https://graph.microsoft.com/.default']


# Azure AD app credentials
def send_email_via_graph_api(email,password):

    # Create a confidential client application
    app = msal.ConfidentialClientApplication(client_id, authority=authority_url, client_credential=client_secret)

    # Acquire a token for the application
    result = app.acquire_token_for_client(scopes=scope)

    if 'access_token' in result:
        access_token = result['access_token']
    else:
        raise Exception(f"Could not obtain access token: {result.get('error_description')}")


    ############ Email Send ##############

    email_addresses = [email,"users@synergy.com"]

    # email_addresses = ['dayanithy.g@synergyship.com', 'nithish.r@synergyship.com']
    for i in email_addresses:
        # The access token obtained from the previous step
        access_token = result['access_token']

        # The email message to send
        email_message_html = {
            "message": {
                "subject": "Your Synergy Analysis Budget Tool Account is Ready!",
                "body": {
                    "contentType": "HTML",
                    "content":  "Welcome to Synergy Analysis!\n\n Your account has been successfully created.\nPassword :"+password+"\n\nThank you!"
                },
                "toRecipients": [
                    {
                        "emailAddress": {"address": f"{i}"}
                        }
                    ],
                # "ccRecipients": get_cc()
            }
        }

        # Set the headers for the request
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        # The Graph API endpoint for sending mail
        send_mail_endpoint = f'https://graph.microsoft.com/v1.0/users/{user_email}/sendMail'

        # Make the POST request to send the email
        response = requests.post(send_mail_endpoint, headers=headers, json=email_message_html)

        if response.status_code == 202:
            print(f"{i} - Email sent successfully")
        else:
            print(f"{i} - Failed to send email: {response.status_code}")
            print(response.json())