import { Navigate, Route, Routes } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "./auth/AuthContext";
import { ChatProvider } from "./chat/ChatContext";
import { AppShell } from "./components/AppShell";
import { AccessPage } from "./pages/AccessPage";
import { ChatPage } from "./pages/ChatPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { LoginPage } from "./pages/LoginPage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  const { user, loading } = useAuth();

  if (loading)
    return (
      <div className="grid h-full place-items-center text-muted" role="status" aria-label="Loading">
        <Loader2 className="size-6 animate-spin" />
      </div>
    );
  if (!user) return <LoginPage />;

  return (
    <ChatProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<ChatPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="access" element={<AccessPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </ChatProvider>
  );
}
