import { Link } from "react-router-dom"

function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-3xl font-bold mb-10 text-center">Image Classification and Similarity Computation</h1>

      <div className="flex flex-wrap justify-center gap-8">
        <Link to="/classification" className="block">
          <div className="w-64 h-64 border-2 border-black rounded-lg cursor-pointer hover:shadow-lg transition-shadow overflow-hidden">
            <div className="w-full h-full flex flex-col items-center justify-center bg-emerald-300 p-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-16 w-16 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span className="text-xl font-medium">Image Classification</span>
              <p className="text-sm text-center mt-2">Upload and classify your images</p>
            </div>
          </div>
        </Link>

        <Link to="/similarity" className="block">
          <div className="w-64 h-64 border-2 border-black rounded-lg cursor-pointer hover:shadow-lg transition-shadow overflow-hidden">
            <div className="w-full h-full flex flex-col items-center justify-center bg-sky-400 p-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-16 w-16 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <span className="text-xl font-medium">Similarity Computation</span>
              <p className="text-sm text-center mt-2">Perform similarity comparison between images</p>
            </div>
          </div>
        </Link>

        <Link to="/pdf-similarity" className="block">
          <div className="w-64 h-64 border-2 border-black rounded-lg cursor-pointer hover:shadow-lg transition-shadow overflow-hidden">
            <div className="w-full h-full flex flex-col items-center justify-center bg-sky-400 p-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-16 w-16 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <span className="text-xl font-medium">Similarity Computation</span>
              <p className="text-sm text-center mt-2">Perform similarity comparison between images</p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  )
}

export default HomePage
