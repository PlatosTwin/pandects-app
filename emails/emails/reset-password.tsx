import {
  Body,
  Button,
  Container,
  Head,
  Heading,
  Hr,
  Html,
  Link,
  Preview,
  Section,
  Text,
} from '@react-email/components';

export interface ResetPasswordProps {
  RESET_URL: string;
  YEAR: string;
}

export const ResetPassword = ({ RESET_URL, YEAR }: ResetPasswordProps) => {
  const previewText = 'Reset your password';

  return (
    <Html>
      <Head />
      <Preview>{previewText}</Preview>
      <Body style={main}>
        <Container style={container}>
          <Section style={brandSection}>
            <Text style={brandText}>Pandects</Text>
          </Section>

          <Heading style={title}>Reset your password</Heading>
          <Text style={copy}>
            We received a request to reset your password. Use the button below
            to choose a new one.
          </Text>

          <Section style={buttonSection}>
            <Button href={RESET_URL} style={button}>
              Reset password
            </Button>
          </Section>

          <Text style={note}>
            If the button doesn&apos;t work, copy and paste this URL into your
            browser:
          </Text>
          <Section style={linkSection}>
            <Link href={RESET_URL} style={link}>
              {RESET_URL}
            </Link>
          </Section>

          <Hr style={divider} />

          <Text style={note}>
            If you didn&apos;t request this email, you can safely ignore it.
            Questions? Email{' '}
            <Link
              href="mailto:nmbogdan@alumni.stanford.edu"
              style={emailLink}
            >
              nmbogdan@alumni.stanford.edu
            </Link>
            .
          </Text>

          <Text style={footer}>© {YEAR} Pandects</Text>
        </Container>
      </Body>
    </Html>
  );
};

ResetPassword.PreviewProps = {
  RESET_URL: '{{{RESET_URL}}}',
  YEAR: '{{{YEAR}}}',
} as ResetPasswordProps;

export default ResetPassword;

const main = {
  backgroundColor: '#fbfaf5',
  fontFamily:
    'ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif',
  padding: '12px',
};

const container = {
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0',
  margin: '44px auto',
  maxWidth: '520px',
  padding: '32px',
};

const brandSection = {
  marginBottom: '20px',
};

const brandText = {
  color: '#0f172a',
  fontSize: '14px',
  fontWeight: 600,
  margin: '0',
};

const title = {
  color: '#0f172a',
  fontSize: '22px',
  fontWeight: 600,
  letterSpacing: '-0.01em',
  margin: '0 0 10px',
};

const copy = {
  color: '#334155',
  fontSize: '14px',
  lineHeight: '22px',
  margin: '0',
};

const buttonSection = {
  margin: '24px 0',
  textAlign: 'center' as const,
};

const button = {
  backgroundColor: '#2563eb',
  color: '#ffffff',
  display: 'inline-block',
  fontSize: '14px',
  fontWeight: 600,
  padding: '12px 20px',
  textDecoration: 'none',
};

const note = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  margin: '0',
};

const linkSection = {
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  marginTop: '10px',
  padding: '10px 12px',
};

const link = {
  color: '#0f172a',
  display: 'block',
  fontFamily: 'ui-monospace,SFMono-Regular,Menlo,Monaco,monospace',
  fontSize: '12px',
  lineHeight: '18px',
  textDecoration: 'none',
  wordBreak: 'break-all' as const,
};

const emailLink = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  textDecoration: 'underline',
};

const divider = {
  borderColor: '#e2e8f0',
  margin: '24px 0',
};

const footer = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  margin: '16px 0 0',
};
