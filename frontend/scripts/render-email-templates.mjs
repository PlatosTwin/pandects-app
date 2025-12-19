import fs from "node:fs";
import path from "node:path";

function writeFile(filePath, contents) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, contents, "utf-8");
}

function main() {
  const verifyUrl = "{{VERIFY_URL}}";
  const year = "{{YEAR}}";
  const preheader = "{{PREHEADER}}";
  const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="x-apple-disable-message-reformatting" />
    <meta http-equiv="x-ua-compatible" content="ie=edge" />
    <title>Verify your email</title>
    <style>
      body,
      table,
      td,
      a {
        -webkit-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
      }
      table,
      td {
        mso-table-lspace: 0pt;
        mso-table-rspace: 0pt;
      }
      img {
        -ms-interpolation-mode: bicubic;
      }
      a[x-apple-data-detectors] {
        color: inherit !important;
        text-decoration: none !important;
      }
      @media screen and (max-width: 600px) {
        .container {
          width: 100% !important;
        }
        .px {
          padding-left: 16px !important;
          padding-right: 16px !important;
        }
        .card {
          padding: 20px !important;
        }
        .h1 {
          font-size: 20px !important;
        }
        .btn a {
          display: block !important;
          width: 100% !important;
          box-sizing: border-box !important;
          text-align: center !important;
        }
      }
    </style>
  </head>
  <body
    style="
      margin: 0;
      padding: 0;
      background: #faf9f5;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial,
        'Apple Color Emoji', 'Segoe UI Emoji';
      color: #0f172a;
    "
  >
    <div style="display: none; max-height: 0; overflow: hidden; opacity: 0">
      ${preheader}
    </div>
    <div style="display: none; max-height: 0; overflow: hidden; opacity: 0">
      &nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;
    </div>

    <table
      role="presentation"
      width="100%"
      cellpadding="0"
      cellspacing="0"
      border="0"
      bgcolor="#FAF9F5"
      style="border-collapse: collapse"
    >
      <tr>
        <td align="center" class="px" style="padding: 32px 24px">
          <!--[if (mso)|(IE)]>
            <table role="presentation" width="560" align="center" cellpadding="0" cellspacing="0" border="0">
              <tr>
                <td>
          <![endif]-->

          <table
            role="presentation"
            class="container"
            align="center"
            width="560"
            cellpadding="0"
            cellspacing="0"
            border="0"
            style="width: 560px; max-width: 560px; margin: 0 auto; border-collapse: collapse"
          >
            <tr>
              <td style="padding: 0 4px 16px 4px">
                <div
                  style="
                    font-size: 14px;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    color: #2563eb;
                    font-weight: 700;
                  "
                >
                  Pandects
                </div>
              </td>
            </tr>

            <tr>
              <td
                class="card"
                bgcolor="#FFFFFF"
                style="
                  background: #ffffff;
                  border: 1px solid #e5e7eb;
                  border-radius: 14px;
                  padding: 28px;
                "
              >
                <div class="h1" style="font-size: 22px; font-weight: 800; margin: 0 0 10px 0">
                  Verify your email
                </div>
                <div style="font-size: 15px; line-height: 1.6; color: #334155; margin: 0 0 18px 0">
                  Thanks for creating a Pandects account. Please verify your email address to
                  activate your account, unlock API keys, and access full search results.
                </div>

                <!--[if mso]>
                  <v:roundrect
                    xmlns:v="urn:schemas-microsoft-com:vml"
                    xmlns:w="urn:schemas-microsoft-com:office:word"
                    href="${verifyUrl}"
                    style="height: 44px; v-text-anchor: middle; width: 220px"
                    arcsize="20%"
                    strokecolor="#2563EB"
                    fillcolor="#2563EB"
                  >
                    <w:anchorlock />
                    <center style="color: #ffffff; font-family: Segoe UI, Arial, sans-serif; font-size: 15px; font-weight: 700">
                      Verify email
                    </center>
                  </v:roundrect>
                <![endif]-->
                <!--[if !mso]><!-->
                <table
                  role="presentation"
                  cellpadding="0"
                  cellspacing="0"
                  border="0"
                  class="btn"
                  style="border-collapse: collapse"
                >
                  <tr>
                    <td
                      align="left"
                      bgcolor="#2563EB"
                      style="border-radius: 10px; background: #2563eb"
                    >
                      <a
                        href="${verifyUrl}"
                        style="
                          display: inline-block;
                          background: #2563eb;
                          color: #ffffff;
                          text-decoration: none;
                          font-weight: 700;
                          font-size: 15px;
                          padding: 12px 16px;
                          border-radius: 10px;
                        "
                        >Verify email</a
                      >
                    </td>
                  </tr>
                </table>
                <!--<![endif]-->

                <div style="font-size: 13px; line-height: 1.6; color: #64748b; margin: 18px 0 0 0">
                  If the button doesn’t work, copy and paste this link into your browser:
                </div>
                <div
                  style="
                    margin: 10px 0 0 0;
                    padding: 12px 14px;
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono',
                      'Courier New', monospace;
                    font-size: 12px;
                    line-height: 1.5;
                    color: #0f172a;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                  "
                >
                  ${verifyUrl}
                </div>

                <div style="height: 18px"></div>

                <div style="font-size: 13px; line-height: 1.6; color: #64748b; margin: 0">
                  If you didn’t create a Pandects account, you can ignore this email.
                </div>
              </td>
            </tr>

            <tr>
              <td style="padding: 16px 4px 0 4px">
                <div style="font-size: 12px; line-height: 1.6; color: #94a3b8">
                  © ${year} Pandects. All rights reserved.
                </div>
              </td>
            </tr>
          </table>

          <!--[if (mso)|(IE)]>
                </td>
              </tr>
            </table>
          <![endif]-->
        </td>
      </tr>
    </table>
  </body>
</html>\n`;

  const text = [
    "Verify your email",
    "",
    "Thanks for creating a Pandects account. Please verify your email address to activate your account, unlock API keys, and access full search results.",
    "",
    `Verify: ${verifyUrl}`,
    "",
    "If you didn’t create a Pandects account, you can ignore this email.",
    "",
  ].join("\n");

  const backendTemplatesDir = path.resolve(process.cwd(), "../backend/email_templates");
  writeFile(path.join(backendTemplatesDir, "verify_email.html"), html);
  writeFile(path.join(backendTemplatesDir, "verify_email.txt"), text);
}

main();
